"""
End-to-End 10-Seed Experimental Runner & Comparator Sweep for DAWIN
===================================================================
Orchestrates validation-only hyperparameter/checkpoint selection, 10-seed
training (seeds 101-110) across DAWIN and all parameter-matched comparators
(ReLU-MLP, Time-Transformer, Neural CDE, S5, Mamba, GRU-GNN), RQ3 dt
interventions, RQ4 GPU throughput/memory profiling, and unified multi-format
reporting (.json, .csv, .md) with configurable NVIDIA B200 mixed-precision
policies (--precision {fp32, bf16, fp16}).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dawin_core import compute_dawin_laplacian_loss
from dawin_diagnostics import (
    evaluate_dt_interventions,
    holm_bonferroni,
    paired_bootstrap_and_sign_flip,
    profile_inference_efficiency,
)
from dawin_models import (
    DAWINGNN,
    DAWINMLP,
    DAWINTransformer,
    MatchedGRUGNN,
    MatchedMamba,
    MatchedNeuralCDE,
    MatchedReLUMLP,
    MatchedS5,
    MatchedTimeTransformer,
    count_trainable_parameters,
    find_matched_width,
)

try:
    from dawin_datasets import (
        compute_binary_auroc_auprc,
        compute_masked_imputation_mse,
        compute_trajectory_ade_fde,
        get_benchmark_bundle,
    )
    HAS_BENCHMARK_DATASETS = True
except ImportError:
    HAS_BENCHMARK_DATASETS = False


# ==============================================================================
# 1. Precision & NVIDIA B200 Runtime Policy
# ==============================================================================

def configure_precision_policy(precision: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """
    Configures PyTorch backend precision flags according to the NVIDIA B200
    experimental policy:
      - Matrix-multiplication TF32 disabled (strict FP32 accumulation in matmuls)
      - cuDNN TF32 enabled when CUDA is active
      - Autocast dtype mapped to None (fp32), torch.bfloat16 (bf16), or
        torch.float16 (fp16) with GradScaler enabled exclusively for fp16.
    """
    precision = precision.lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from fp32, bf16, fp16.")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True

    if device.type != "cuda":
        if precision != "fp32":
            print(
                f"[Warning] --precision {precision} requested on {device.type}; "
                "falling back to fp32."
            )
        return None, False

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support bfloat16 autocast.")
        return torch.bfloat16, False
    if precision == "fp16":
        return torch.float16, True
    return None, False


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


# ==============================================================================
# 2. Synthetic & Benchmark Data Loaders
# ==============================================================================

def generate_synthetic_irregular_data(
    task: str = "damped_oscillator",
    n_train: int = 2048,
    n_val: int = 512,
    n_test: int = 512,
    seq_len: int = 32,
    d_in: int = 6,
    d_out: int = 6,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """
    Generates controlled irregular physical dynamics ('damped_oscillator') or
    the time-independent control ('dt_irrelevant') with out-of-range test dt.
    """
    rng = np.random.default_rng(seed)
    total = n_train + n_val + n_test

    x = rng.normal(0.0, 1.0, size=(total, seq_len, d_in)).astype(np.float32)
    dt = rng.uniform(0.05, 0.50, size=(total, seq_len)).astype(np.float32)
    dt[:, 0] = 0.0
    # Out-of-range test intervals (up to 1.25s vs. 0.50s training ceiling)
    dt[n_train + n_val :, 1:] = rng.uniform(0.05, 1.25, size=(n_test, seq_len - 1)).astype(np.float32)

    y = np.zeros((total, seq_len, d_out), dtype=np.float32)
    w_proj = rng.normal(0.0, 0.5, size=(d_in, d_out)).astype(np.float32)

    if task == "damped_oscillator":
        omega = np.linspace(1.5, 4.5, d_out, dtype=np.float32)
        gamma = np.linspace(0.25, 0.85, d_out, dtype=np.float32)
        state = np.zeros((total, d_out), dtype=np.float32)
        for t in range(seq_len):
            drive = np.tanh(x[:, t] @ w_proj)
            decay = np.exp(-gamma[None, :] * dt[:, t : t + 1])
            rot = np.cos(omega[None, :] * dt[:, t : t + 1])
            state = decay * rot * state + drive * (1.0 - np.exp(-dt[:, t : t + 1]))
            y[:, t] = state
    elif task == "dt_irrelevant":
        for t in range(seq_len):
            prev = x[:, t - 1] if t > 0 else np.zeros_like(x[:, t])
            y[:, t] = np.sin(x[:, t] @ w_proj) + 0.5 * np.tanh(prev @ w_proj)
    else:
        raise ValueError(f"Unknown synthetic task '{task}'.")

    # Training-only z-score normalization
    y_train = y[:n_train]
    mean = y_train.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(y_train.std(axis=(0, 1), keepdims=True), 1e-6)
    y_norm = (y - mean) / std

    def _make_loader(sl: slice, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(x[sl]),
            torch.from_numpy(dt[sl]),
            torch.from_numpy(y_norm[sl]),
        )
        return DataLoader(ds, batch_size=256, shuffle=shuffle)

    meta = {
        "d_in": d_in,
        "d_out": d_out,
        "seq_len": seq_len,
        "task_type": "regression",
        "metric_name": "NRMSE",
        "higher_is_better": False,
    }
    return (
        _make_loader(slice(0, n_train), True),
        _make_loader(slice(n_train, n_train + n_val), False),
        _make_loader(slice(n_train + n_val, total), False),
        meta,
    )


def _unpack_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        x = batch["x"].to(device)
        dt = batch["dt"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device) if "mask" in batch and batch["mask"] is not None else None
        return x, dt, y, mask
    x, dt, y = batch
    return x.to(device), dt.to(device), y.to(device), None


# ==============================================================================
# 3. Model & Comparator Factory with Parameter-Parity Solver
# ==============================================================================

SEQUENCE_COMPARATORS = ["relu_mlp", "time_transformer", "neural_cde", "s5", "mamba"]


def build_dawin_model(
    backbone: str,
    d_in: int,
    d_out: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    ablation: str = "none",
    use_parallel_scan: bool = True,
) -> nn.Module:
    if backbone == "mlp":
        return DAWINMLP(
            in_features=d_in,
            out_features=d_out,
            hidden_width=hidden_dim,
            num_layers=num_layers,
            ablation=ablation,
            use_parallel_scan=use_parallel_scan,
        )
    if backbone == "transformer":
        return DAWINTransformer(
            in_features=d_in,
            out_features=d_out,
            d_model=hidden_dim,
            nhead=4,
            num_layers=num_layers,
            ablation=ablation,
            use_parallel_scan=use_parallel_scan,
        )
    if backbone == "gnn":
        return DAWINGNN(
            in_features=d_in,
            out_features=d_out,
            hidden_width=hidden_dim,
            num_layers=num_layers,
            ablation=ablation,
            use_parallel_scan=use_parallel_scan,
        )
    raise ValueError(f"Unsupported backbone '{backbone}'.")


def build_matched_comparator(
    comp_name: str,
    d_in: int,
    d_out: int,
    target_params: int,
    num_layers: int = 2,
) -> tuple[nn.Module, int, float]:
    """
    Constructs a parameter-matched baseline within <= 2% of target_params.
    """
    builders: dict[str, Callable[[int], nn.Module]] = {
        "relu_mlp": lambda w: MatchedReLUMLP(d_in, d_out, hidden_width=w, num_layers=num_layers),
        "time_transformer": lambda w: MatchedTimeTransformer(
            d_in, d_out, d_model=max(16, (w // 2) * 2), nhead=2, num_layers=num_layers
        ),
        "neural_cde": lambda w: MatchedNeuralCDE(d_in, d_out, hidden_width=w),
        "s5": lambda w: MatchedS5(d_in, d_out, d_model=w, num_layers=num_layers),
        "mamba": lambda w: MatchedMamba(d_in, d_out, d_model=w, num_layers=num_layers),
        "gru_gnn": lambda w: MatchedGRUGNN(d_in, d_out, hidden_width=w, num_layers=num_layers),
    }
    if comp_name not in builders:
        raise ValueError(f"Unknown comparator '{comp_name}'.")

    width, _, _ = find_matched_width(
        target_params,
        builders[comp_name],
        min_width=16,
        max_width=1024,
        max_rel_diff=0.025,
    )
    model = builders[comp_name](width)
    actual_params = count_trainable_parameters(model)
    rel_diff_pct = 100.0 * abs(actual_params - target_params) / max(target_params, 1)
    return model, width, rel_diff_pct


# ==============================================================================
# 4. Training & Evaluation Loop
# ==============================================================================

@dataclass
class RunMetrics:
    model_name: str
    seed: int
    precision: str
    params: int
    rel_param_diff_pct: float
    best_epoch: int
    val_metric: float
    test_metric: float
    dt_shuffled_delta: float
    dt_constant_delta: float
    throughput_transitions_s: float
    peak_mem_mib: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss_and_metric(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None,
    task_type: str,
) -> tuple[torch.Tensor, float]:
    if task_type == "binary_classification":
        # Sequence-level logit from final step
        logits = pred[:, -1, 0] if pred.dim() == 3 else pred.squeeze(-1)
        targets = y.float().view_as(logits)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return loss, float(loss.item())

    if task_type == "masked_imputation":
        if mask is None:
            mask = torch.ones_like(y)
        sq_err = ((pred - y) ** 2) * mask
        loss = sq_err.sum() / mask.sum().clamp_min(1.0)
        return loss, float(loss.item())

    if task_type == "trajectory" and pred.dim() == 3 and y.dim() == 3 and pred.shape != y.shape:
        pred = pred[:, -1, :].view_as(y)

    # Default: sequence regression / trajectory displacement (MSE training, NRMSE eval)
    mse = nn.functional.mse_loss(pred, y)
    nrmse = float(torch.sqrt(mse).item())
    return mse, nrmse


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    task_type: str,
) -> float:
    model.eval()
    if task_type == "binary_classification" and HAS_BENCHMARK_DATASETS:
        all_logits, all_targets = [], []
        for batch in loader:
            x, dt, y, _ = _unpack_batch(batch, device)
            with autocast_context(device, amp_dtype):
                pred = model(x, dt)
            logits = pred[:, -1, 0] if pred.dim() == 3 else pred.squeeze(-1)
            all_logits.append(logits.float().cpu())
            all_targets.append(y.float().view_as(logits).cpu())
        auroc, _ = compute_binary_auroc_auprc(torch.cat(all_logits), torch.cat(all_targets))
        return float(auroc)

    total_sq_err = 0.0
    total_count = 0.0
    for batch in loader:
        x, dt, y, mask = _unpack_batch(batch, device)
        with autocast_context(device, amp_dtype):
            pred = model(x, dt)
        pred_f = pred.float()
        y_f = y.float()
        if task_type == "trajectory" and pred_f.dim() == 3 and y_f.dim() == 3 and pred_f.shape != y_f.shape:
            pred_f = pred_f[:, -1, :].view_as(y_f)
        if task_type == "masked_imputation" and mask is not None:
            m_f = mask.float()
            total_sq_err += float((((pred_f - y_f) ** 2) * m_f).sum().item())
            total_count += float(m_f.sum().clamp_min(1.0).item())
        else:
            total_sq_err += float(((pred_f - y_f) ** 2).sum().item())
            total_count += float(y_f.numel())

    mse = total_sq_err / max(total_count, 1.0)
    return mse if task_type == "masked_imputation" else math.sqrt(mse)


def train_single_seed(
    model: nn.Module,
    model_name: str,
    rel_param_diff_pct: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    meta: dict[str, Any],
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    precision: str,
    amp_dtype: torch.dtype | None,
    use_scaler: bool,
    device: torch.device,
    laplacian_weight: float = 1e-4,
) -> RunMetrics:
    set_seed(seed)
    model = model.to(device)
    task_type = meta["task_type"]
    higher_is_better = meta["higher_is_better"]

    # Unfused AdamW matching the WSP/DAWIN B200 training specification
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val = -float("inf") if higher_is_better else float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x, dt, y, mask = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_dtype):
                pred = model(x, dt)
                loss, _ = compute_loss_and_metric(pred, y, mask, task_type)
                if model_name == "dawin" and laplacian_weight > 0.0:
                    loss = loss + laplacian_weight * compute_dawin_laplacian_loss(model)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered for {model_name} (seed={seed}, precision={precision})."
                )

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        scheduler.step()
        val_score = evaluate_loader(model, val_loader, device, amp_dtype, task_type)
        improved = (val_score > best_val) if higher_is_better else (val_score < best_val)

        if improved:
            best_val = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_score = evaluate_loader(model, test_loader, device, amp_dtype, task_type)

    # RQ3: Elapsed-time intervention diagnostics (regression / displacement tasks)
    dt_shuf_delta = 0.0
    dt_const_delta = 0.0
    if task_type in {"regression", "trajectory"}:
        with autocast_context(device, amp_dtype):
            dt_res = evaluate_dt_interventions(model, test_loader, device)
        dt_shuf_delta = float(dt_res.get("delta_shuffled", dt_res.get("shuffled_delta", 0.0)))
        dt_const_delta = float(dt_res.get("delta_constant", dt_res.get("constant_delta", 0.0)))

    # RQ4: Inference throughput & peak GPU memory under active precision policy
    seq_len = int(meta["seq_len"])
    meta["laplacian_weight"] = float(laplacian_weight)
    d_in = int(meta["d_in"])
    with autocast_context(device, amp_dtype):
        prof = profile_inference_efficiency(
            model=model,
            batch_size=256,
            seq_len=seq_len,
            in_features=d_in,
            device=device,
            dtype=amp_dtype,
            warmup_iters=5,
            timed_iters=20,
        )

    return RunMetrics(
        model_name=model_name,
        seed=seed,
        precision=precision,
        params=count_trainable_parameters(model),
        rel_param_diff_pct=rel_param_diff_pct,
        best_epoch=best_epoch,
        val_metric=best_val,
        test_metric=test_score,
        dt_shuffled_delta=dt_shuf_delta,
        dt_constant_delta=dt_const_delta,
        throughput_transitions_s=float(prof["throughput_transitions_s"]),
        peak_mem_mib=float(prof["peak_mem_mib"]),
    )


# ==============================================================================
# 5. Multi-Comparator Sweep & Unified Reporting
# ==============================================================================

def summarize_and_save_reports(
    all_runs: dict[str, list[RunMetrics]],
    meta: dict[str, Any],
    out_dir: str,
    prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dawin_runs = all_runs["dawin"]
    dawin_scores = np.array([r.test_metric for r in dawin_runs], dtype=np.float64)
    higher_is_better = meta["higher_is_better"]

    comparator_names = [k for k in all_runs if k != "dawin"]
    raw_pvals: list[float] = []
    stats_by_comp: dict[str, dict[str, Any]] = {}

    for comp in comparator_names:
        comp_scores = np.array([r.test_metric for r in all_runs[comp]], dtype=np.float64)
        # Positive delta always favors DAWIN
        diffs = (dawin_scores - comp_scores) if higher_is_better else (comp_scores - dawin_scores)
        res = paired_bootstrap_and_sign_flip(diffs, np.zeros_like(diffs), n_resamples=20000, seed=42)
        wins = int(np.sum(diffs > 0))
        stats_by_comp[comp] = {
            "delta_mean": float(np.mean(diffs)),
            "ci_low": float(res["ci_low"]),
            "ci_high": float(res["ci_high"]),
            "p_raw": float(res.get("p_value", res["exact_p"])),
            "dawin_wins": f"{wins}/{len(diffs)}",
        }
        raw_pvals.append(float(res.get("p_value", res["exact_p"])))

    adj_pvals = holm_bonferroni(raw_pvals) if raw_pvals else []
    for comp, p_adj in zip(comparator_names, adj_pvals):
        stats_by_comp[comp]["p_holm"] = float(p_adj)

    rows: list[dict[str, Any]] = []
    for name, runs in all_runs.items():
        vals = np.array([r.val_metric for r in runs], dtype=np.float64)
        tests = np.array([r.test_metric for r in runs], dtype=np.float64)
        shufs = np.array([r.dt_shuffled_delta for r in runs], dtype=np.float64)
        consts = np.array([r.dt_constant_delta for r in runs], dtype=np.float64)
        thr = np.array([r.throughput_transitions_s for r in runs], dtype=np.float64)
        mem = np.array([r.peak_mem_mib for r in runs], dtype=np.float64)

        row: dict[str, Any] = {
            "model": name,
            "precision": runs[0].precision,
            "params": runs[0].params,
            "param_diff_pct": round(runs[0].rel_param_diff_pct, 3),
            "val_mean": float(np.mean(vals)),
            "val_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "test_mean": float(np.mean(tests)),
            "test_sd": float(np.std(tests, ddof=1)) if len(tests) > 1 else 0.0,
            "dt_shuf_delta": float(np.mean(shufs)),
            "dt_const_delta": float(np.mean(consts)),
            "throughput_trans_s": float(np.mean(thr)),
            "peak_mem_mib": float(np.mean(mem)),
        }
        if name == "dawin":
            row.update({"delta_vs_dawin": 0.0, "ci_95": "—", "holm_p": "—", "dawin_wins": "—"})
        else:
            st = stats_by_comp[name]
            row.update(
                {
                    "delta_vs_dawin": st["delta_mean"],
                    "ci_95": f"[{st['ci_low']:.5f}, {st['ci_high']:.5f}]",
                    "holm_p": f"{st['p_holm']:.4f}",
                    "dawin_wins": st["dawin_wins"],
                }
            )
        rows.append(row)

    # 1. Save JSON
    json_path = os.path.join(out_dir, f"{prefix}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "summary": rows,
                "per_seed": {k: [asdict(r) for r in v] for k, v in all_runs.items()},
            },
            f,
            indent=2,
        default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o),
        )

    # 2. Save CSV
    csv_path = os.path.join(out_dir, f"{prefix}_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 3. Save Markdown Table
    md_path = os.path.join(out_dir, f"{prefix}_table.md")
    metric_label = meta.get("metric_name", meta.get("metric", "score"))
    lines = [
        f"### DAWIN Unified Comparator Sweep (`{prefix}` | Precision: `{rows[0]['precision'].upper()}`)",
        "",
        f"| Model | Params (Δ%) | Val {metric_label} | Test {metric_label} | Δ vs DAWIN (95% CI) | Holm p | Wins | Δt Shuf Δ | Throughput (trans/s) | Peak MiB |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| **{r['model']}** | {r['params']:,} ({r['param_diff_pct']:.2f}%) | "
            f"{r['val_mean']:.5f} ± {r['val_sd']:.5f} | "
            f"{r['test_mean']:.5f} ± {r['test_sd']:.5f} | "
            f"{r['delta_vs_dawin']:+.5f} {r['ci_95']} | "
            f"{r['holm_p']} | {r['dawin_wins']} | "
            f"{r['dt_shuf_delta']:+.5f} | {r['throughput_trans_s']:,.0f} | {r['peak_mem_mib']:.2f} |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nSaved JSON, CSV, and Markdown reports to {out_dir}/{prefix}_*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10-seed DAWIN experiments and comparator sweeps.")
    parser.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "transformer", "gnn"])
    parser.add_argument(
        "--task",
        type=str,
        default="damped_oscillator",
        help="Synthetic task ('damped_oscillator', 'dt_irrelevant') or benchmark name "
        "('mujoco_halfcheetah', 'mujoco_hopper', 'mujoco_reacher', 'mujoco_walker2d', "
        "'physionet_mortality', 'physionet_imputation', 'physionet_dense', 'eth_ucy_jaad').",
    )
    parser.add_argument(
        "--comparators",
        nargs="+",
        default=SEQUENCE_COMPARATORS,
        choices=["relu_mlp", "time_transformer", "neural_cde", "s5", "mamba", "gru_gnn"],
        help="Parameter-matched baselines to sweep alongside DAWIN.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Numerical precision policy: fp32, bf16 (B200 default), or fp16.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--laplacian-weight", type=float, default=1e-4)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(101, 111)),
        help="Final evaluation seeds (default: 101..110).",
    )
    parser.add_argument("--ablation", type=str, default="none")
    parser.add_argument("--no-parallel-scan", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results_dawin")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = configure_precision_policy(args.precision, device)
    effective_precision = args.precision if device.type == "cuda" else "fp32"
    print(f"Device: {device} | Requested precision: {args.precision} | Active precision: {effective_precision}")

    # Load dataset bundle
    if args.task in {"damped_oscillator", "dt_irrelevant"}:
        train_loader, val_loader, test_loader, meta = generate_synthetic_irregular_data(task=args.task)
    elif HAS_BENCHMARK_DATASETS:
        bundle = get_benchmark_bundle(args.task)
        train_loader, val_loader, test_loader = bundle.get_loaders(batch_size=256)
        meta = dict(bundle.meta) if bundle.meta else {}
        meta.setdefault("d_in", bundle.in_features)
        meta.setdefault("d_out", bundle.out_features)
        meta.setdefault("seq_len", bundle.seq_len)
        meta.setdefault("pred_len", bundle.pred_len)
        meta.setdefault("task_type", bundle.task_type)
        meta.setdefault("stats", bundle.stats)
        meta.setdefault("higher_is_better", bundle.task_type == "binary_classification")
    else:
        raise RuntimeError(f"Task '{args.task}' requires dawin_datasets.py to be importable.")

    d_in = int(meta["d_in"])
    d_out = int(meta["d_out"])

    # Instantiate reference DAWIN model to lock target parameter budget
    ref_dawin = build_dawin_model(
        backbone=args.backbone,
        d_in=d_in,
        d_out=d_out,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        ablation=args.ablation,
        use_parallel_scan=not args.no_parallel_scan,
    )
    target_params = count_trainable_parameters(ref_dawin)
    print(f"Target DAWIN ({args.backbone}) parameter count: {target_params:,}")

    all_runs: dict[str, list[RunMetrics]] = {"dawin": []}
    for comp in args.comparators:
        all_runs[comp] = []

    # 1. Train DAWIN across all seeds
    for seed in args.seeds:
        model = build_dawin_model(
            backbone=args.backbone,
            d_in=d_in,
            d_out=d_out,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            ablation=args.ablation,
            use_parallel_scan=not args.no_parallel_scan,
        )
        metrics = train_single_seed(
            model=model,
            model_name="dawin",
            rel_param_diff_pct=0.0,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            meta=meta,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            precision=effective_precision,
            amp_dtype=amp_dtype,
            use_scaler=use_scaler,
            device=device,
            laplacian_weight=args.laplacian_weight,
        )
        all_runs["dawin"].append(metrics)
        print(
            f"[dawin | seed={seed} | {effective_precision}] "
            f"val={metrics.val_metric:.5f} | test={metrics.test_metric:.5f} | "
            f"dt_shuf_delta={metrics.dt_shuffled_delta:+.5f}"
        )

    # 2. Sweep all parameter-matched comparators across identical seeds
    for comp in args.comparators:
        _, matched_w, rel_diff = build_matched_comparator(
            comp_name=comp,
            d_in=d_in,
            d_out=d_out,
            target_params=target_params,
            num_layers=args.num_layers,
        )
        print(f"\n--- Comparator: {comp} (matched width={matched_w}, param diff={rel_diff:.2f}%) ---")
        for seed in args.seeds:
            model, _, _ = build_matched_comparator(
                comp_name=comp,
                d_in=d_in,
                d_out=d_out,
                target_params=target_params,
                num_layers=args.num_layers,
            )
            metrics = train_single_seed(
                model=model,
                model_name=comp,
                rel_param_diff_pct=rel_diff,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                meta=meta,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                precision=effective_precision,
                amp_dtype=amp_dtype,
                use_scaler=use_scaler,
                device=device,
            )
            all_runs[comp].append(metrics)
            print(
                f"[{comp} | seed={seed} | {effective_precision}] "
                f"val={metrics.val_metric:.5f} | test={metrics.test_metric:.5f}"
            )

    prefix = f"{args.backbone}_{args.task}_{effective_precision}"
    summarize_and_save_reports(all_runs=all_runs, meta=meta, out_dir=args.out_dir, prefix=prefix)


if __name__ == "__main__":
    main()
