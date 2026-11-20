"""
dawin_diagnostics.py

Diagnostic, statistical, and hardware profiling utilities for the DAWIN paper:
1. RQ3 Elapsed-Time Intervention Suite (`evaluate_dt_interventions`):
   - True elapsed intervals (`dt_true`)
   - Batch-shuffled elapsed intervals (`dt_perm`)
   - Constant batch-mean elapsed intervals (`dt_const`)
   - Zero-interval boundary probe (`dt_zero`)
2. Paired Statistical Testing (`paired_bootstrap_and_sign_flip`, `holm_bonferroni`):
   - 20,000-resample paired percentile bootstrap 95% confidence intervals
   - Exact two-sided paired sign-flip permutation p-values
   - Holm-Bonferroni family-wise multiplicity adjustment
3. RQ4 NVIDIA B200 / GPU Inference & Training Efficiency Profiler (`profile_inference_efficiency`):
   - 30 warmup iterations + 200 synchronized CUDA event passes
   - Measures per-batch latency (ms), throughput (transitions/s), and peak allocated VRAM (MiB)
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_nrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> float:
    """Computes dimensionless Normalized Root-Mean-Square Error (NRMSE)."""
    diff_sq = (pred - target) ** 2
    if mask is not None:
        m = mask.to(dtype=diff_sq.dtype, device=diff_sq.device)
        while m.dim() < diff_sq.dim():
            m = m.unsqueeze(-1)
        if m.shape != diff_sq.shape and m.shape[-1] != diff_sq.shape[-1]:
            m = torch.ones_like(diff_sq)
        else:
            m = m.expand_as(diff_sq)
        denom = torch.clamp(m.sum(), min=1.0)
        mse = (diff_sq * m).sum() / denom
    else:
        mse = diff_sq.mean()
    return float(torch.sqrt(mse + eps).item())


@torch.no_grad()
def evaluate_dt_interventions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], float]] = None,
    seed: int = 101,
) -> Dict[str, float]:
    """
    Evaluates a trained model under controlled elapsed-time (`dt`) perturbations
    while holding model weights, spatial inputs `x`, and ground-truth targets `y` fixed.

    Returns dictionary with:
      - `true`: metric on true intervals
      - `shuffled`: metric under batch-permuted intervals (`dt_perm`)
      - `constant`: metric under batch-mean constant intervals (`dt_const`)
      - `zero`: metric under zero elapsed intervals (`dt_zero = 0`)
      - `delta_shuffled`: `shuffled - true` (positive = degradation for error metrics)
      - `delta_constant`: `constant - true`
      - `rel_pct_shuffled`: `100 * (shuffled - true) / max(|true|, 1e-8)`
      - `rel_pct_constant`: `100 * (constant - true) / max(|true|, 1e-8)`
    """
    model.eval()
    if metric_fn is None:
        metric_fn = compute_nrmse

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    preds_true: List[torch.Tensor] = []
    preds_shuf: List[torch.Tensor] = []
    preds_const: List[torch.Tensor] = []
    preds_zero: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    masks_all: List[Optional[torch.Tensor]] = []

    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch["x"].to(device)
            dt = batch["dt"].to(device)
            y = batch["y"].to(device)
            mask = batch.get("mask", None)
            adj = batch.get("adj", None)
        else:
            it = iter(batch)
            x = next(it).to(device)
            dt = next(it).to(device)
            y = next(it).to(device)
            mask = next(it, None)
            adj = next(it, None)
        if mask is not None:
            mask = mask.to(device)
        if adj is not None:
            adj = adj.to(device)

        def _forward_with_dt(dt_mod: torch.Tensor) -> torch.Tensor:
            out = model(x, dt_mod, adj=adj) if adj is not None else model(x, dt_mod)
            if out.dim() == 3 and y.dim() == 3 and out.shape != y.shape:
                out = out[:, -1, :].view_as(y)
            return out

        # 1. True intervals
        out_true = _forward_with_dt(dt)

        # 2. Batch-shuffled intervals (permutes intervals across batch/time while preserving marginal distribution)
        B = dt.shape[0]
        if B > 1:
            perm_b = torch.randperm(B, generator=gen).to(device)
            dt_shuf = dt[perm_b]
        else:
            L = dt.shape[1]
            perm_l = torch.randperm(L, generator=gen).to(device)
            dt_shuf = dt[:, perm_l]
        out_shuf = _forward_with_dt(dt_shuf)

        # 3. Constant mean interval per batch
        dt_mean = dt.mean(dim=tuple(range(1, dt.dim())), keepdim=True).expand_as(dt)
        out_const = _forward_with_dt(dt_mean)

        # 4. Zero-time boundary probe (dt = 0)
        dt_zero = torch.zeros_like(dt)
        out_zero = _forward_with_dt(dt_zero)

        preds_true.append(out_true.detach().cpu())
        preds_shuf.append(out_shuf.detach().cpu())
        preds_const.append(out_const.detach().cpu())
        preds_zero.append(out_zero.detach().cpu())
        targets_all.append(y.detach().cpu())
        masks_all.append(mask.detach().cpu() if mask is not None else None)

    y_cat = torch.cat(targets_all, dim=0)
    m_cat = torch.cat(masks_all, dim=0) if masks_all[0] is not None else None

    p_true = torch.cat(preds_true, dim=0)
    p_shuf = torch.cat(preds_shuf, dim=0)
    p_const = torch.cat(preds_const, dim=0)
    if p_true.dim() == 3 and y_cat.dim() == 3 and p_true.shape != y_cat.shape:
        p_true = p_true[:, -1, :].view_as(y_cat)
        p_shuf = p_shuf[:, -1, :].view_as(y_cat)
        p_const = p_const[:, -1, :].view_as(y_cat)
    m_eval = None if (m_cat is not None and m_cat.shape[1] != y_cat.shape[1]) else m_cat
    m_true = float(metric_fn(p_true, y_cat, m_eval))
    m_shuf = float(metric_fn(p_shuf, y_cat, m_eval))
    m_const = float(metric_fn(p_const, y_cat, m_eval))
    m_zero = float(metric_fn(torch.cat(preds_zero, dim=0), y_cat, m_cat))

    denom = max(abs(m_true), 1e-8)
    return {
        "true": m_true,
        "shuffled": m_shuf,
        "constant": m_const,
        "zero": m_zero,
        "delta_shuffled": m_shuf - m_true,
        "delta_constant": m_const - m_true,
        "rel_pct_shuffled": 100.0 * (m_shuf - m_true) / denom,
        "rel_pct_constant": 100.0 * (m_const - m_true) / denom,
    }


def paired_bootstrap_and_sign_flip(
    baseline_scores: Sequence[float],
    target_scores: Optional[Sequence[float]] = None,
    n_resamples: int = 20_000,
    ci_level: float = 0.95,
    seed: int = 2026,
    higher_is_better: bool = False,
    n_bootstrap: Optional[int] = None,
) -> Dict[str, Any]:
    if n_bootstrap is not None:
        n_resamples = n_bootstrap
    """
    Computes paired seed-level comparison statistics matching WSP/DAWIN protocol:
      - `delta_mean`: mean(baseline - target) if lower_is_better else mean(target - baseline)
        (positive `delta_mean` always favors `target` / DAWIN)
      - `ci_low`, `ci_high`: 20,000-resample paired percentile bootstrap confidence interval
      - `exact_p`: exact two-sided paired sign-flip permutation p-value (for n <= 16)
      - `wins`: count of seeds where target strictly beats baseline
      - `n_seeds`: number of paired seeds
    """
    b_arr = np.asarray(baseline_scores, dtype=np.float64)
    t_arr = np.zeros_like(b_arr) if target_scores is None else np.asarray(target_scores, dtype=np.float64)
    assert b_arr.shape == t_arr.shape and b_arr.ndim == 1, "Scores must be 1D arrays of equal length"
    n = len(b_arr)
    assert n >= 1, "At least one paired seed is required"

    diffs = (t_arr - b_arr) if higher_is_better else (b_arr - t_arr)
    delta_mean = float(np.mean(diffs))
    delta_sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    wins = int(np.sum(diffs > 0.0))

    # 20,000-resample paired percentile bootstrap
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(diffs[indices], axis=1)
    alpha = 1.0 - ci_level
    ci_low = float(np.quantile(boot_means, alpha / 2.0))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    # Exact two-sided sign-flip test
    obs_abs = abs(delta_mean)
    if n <= 16:
        signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=np.float64)
        perm_means = np.abs(np.mean(signs * diffs[None, :], axis=1))
        exact_p = float(np.mean(perm_means >= obs_abs - 1e-15))
    else:
        rand_signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
        perm_means = np.abs(np.mean(rand_signs * diffs[None, :], axis=1))
        exact_p = float((np.sum(perm_means >= obs_abs - 1e-15) + 1) / (n_resamples + 1))

    return {
        "delta_mean": delta_mean,
        "delta_sd": delta_sd,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "exact_p": exact_p,
        "wins": wins,
        "n_seeds": n,
    }


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down family-wise error rate adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        mult = m - rank
        cand = min(1.0, float(p_val) * mult)
        running_max = max(running_max, cand)
        adjusted[orig_idx] = running_max
    return adjusted


def profile_inference_efficiency(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    in_features: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 30,
    timed_iters: int = 200,
    include_backward: bool = False,
    adj: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Profiles inference (and optional forward+backward) latency, transition throughput,
    and peak allocated GPU memory following the WSP/DAWIN B200 protocol:
      - Resident synthetic inputs (`x`, `dt`) pre-allocated on target device
      - 30 warmup iterations + 200 synchronized iterations
      - Reports per-batch latency (ms), throughput (transitions/s), and peak allocated MiB
    """
    is_cuda = device.type == "cuda"
    model = model.to(device)
    if include_backward:
        model.train()
    else:
        model.eval()

    x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.float32)
    dt = torch.rand(batch_size, seq_len, device=device, dtype=torch.float32) * 0.5 + 0.01

    if adj is not None:
        adj = adj.to(device)

    use_amp = is_cuda and dtype in (torch.bfloat16, torch.float16)

    def _step() -> None:
        if include_backward:
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                out = model(x, dt, adj=adj) if adj is not None else model(x, dt)
                loss = out.float().square().mean()
            loss.backward()
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    _ = model(x, dt, adj=adj) if adj is not None else model(x, dt)

    # Warmup passes
    for _ in range(warmup_iters):
        _step()

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(timed_iters):
            _step()
        end_event.record()
        torch.cuda.synchronize(device)

        total_ms = float(start_event.elapsed_time(end_event))
        peak_bytes = float(torch.cuda.max_memory_allocated(device))
        peak_mib = peak_bytes / (1024.0 * 1024.0)
    else:
        t0 = time.perf_counter()
        for _ in range(timed_iters):
            _step()
        total_ms = (time.perf_counter() - t0) * 1000.0
        peak_mib = 0.0

    latency_ms = total_ms / max(timed_iters, 1)
    total_transitions = float(batch_size * seq_len)
    transitions_per_sec = total_transitions / max(latency_ms / 1000.0, 1e-12)

    return {
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "latency_ms": latency_ms,
        "throughput_transitions_s": transitions_per_sec,
        "peak_mem_mib": peak_mib,
        "transitions_per_sec": transitions_per_sec,
        "peak_memory_mib": peak_mib,
        "include_backward": bool(include_backward),
    }
