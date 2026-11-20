"""
Automated Multi-Task Batch Runner & RQ4 B200 GPU Profiler for DAWIN
=================================================================
Sequentially executes all 10-seed (101..110) parameter-matched sweeps across:
  1. Synthetic controls (damped_oscillator, dt_irrelevant)
  2. MuJoCo irregular physical dynamics (HalfCheetah, Hopper, Reacher, Walker2d)
  3. PhysioNet Challenge 2012 (mask-aware mortality, mask-aware imputation, dense WSP)
  4. ETH/UCY + JAAD pedestrian trajectories (position-only, full motion features)

Also supports `--profile-b200` / `--profile-only` to generate RQ4 GPU throughput
and peak memory scaling tables across batch sizes {1, 16, 256, 4096} and sequence
lengths {32, 512, 4096, 16384}, and automatically compiles publication-ready
LaTeX tables via `export_dawin_latex_tables.py` upon completion.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from dawin_models import (
    DAWINMLP,
    DAWINTransformer,
    MatchedMamba,
    MatchedNeuralCDE,
    MatchedReLUMLP,
    MatchedS5,
    MatchedTimeTransformer,
    count_trainable_parameters,
    find_matched_width,
)


@dataclass
class BenchmarkJob:
    suite: str
    task: str
    backbone: str
    metric_name: str
    higher_is_better: bool
    default_epochs: int
    default_batch_size: int
    default_width: int = 128
    default_layers: int = 2


BENCHMARK_MANIFEST: List[BenchmarkJob] = [
    BenchmarkJob("synthetic", "damped_oscillator", "mlp", "NRMSE", False, 100, 256, 128, 2),
    BenchmarkJob("synthetic", "dt_irrelevant", "mlp", "NRMSE", False, 100, 256, 128, 2),
    BenchmarkJob("mujoco", "HalfCheetah-v5", "mlp", "NRMSE", False, 100, 256, 128, 2),
    BenchmarkJob("mujoco", "Hopper-v5", "mlp", "NRMSE", False, 100, 256, 128, 2),
    BenchmarkJob("mujoco", "Reacher-v5", "mlp", "NRMSE", False, 100, 256, 128, 2),
    BenchmarkJob("mujoco", "Walker2d-v5", "mlp", "NRMSE", False, 100, 256, 128, 2),
    BenchmarkJob("physionet", "PhysioNet2012-mortality", "transformer", "AUROC", True, 60, 128, 128, 2),
    BenchmarkJob("physionet", "PhysioNet2012-imputation", "transformer", "Masked_MSE", False, 60, 128, 128, 2),
    BenchmarkJob("physionet", "PhysioNet2012-dense", "transformer", "NRMSE", False, 80, 256, 128, 2),
    BenchmarkJob("trajectories", "ETH_UCY_JAAD-position", "mlp", "ADE", False, 80, 256, 128, 2),
    BenchmarkJob("trajectories", "ETH_UCY_JAAD-motion", "mlp", "ADE", False, 80, 256, 128, 2),
]


def configure_b200_runtime_policy(precision: str) -> Optional[torch.dtype]:
    """Configures B200 runtime precision flags matching Section 5."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _build_b200_profile_models(
    d_in: int = 32,
    d_out: int = 16,
    ref_width: int = 128,
    num_layers: int = 2,
) -> Dict[str, nn.Module]:
    """Constructs DAWIN and parameter-matched comparators for RQ4 B200 profiling."""
    def _w(res):
        return int(res[0]) if isinstance(res, tuple) else int(res)

    ref_mlp = DAWINMLP(d_in, d_out, ref_width, num_layers=num_layers, use_parallel_scan=True)
    target_params = count_trainable_parameters(ref_mlp)

    ref_serial = DAWINMLP(d_in, d_out, ref_width, num_layers=num_layers, use_parallel_scan=False)
    dawin_tf = DAWINTransformer(d_in, d_out, ref_width, num_layers=num_layers, use_parallel_scan=True)

    w_relu = _w(find_matched_width(target_params, lambda w: MatchedReLUMLP(d_in, d_out, w, num_layers=num_layers)))
    w_tf = _w(find_matched_width(target_params, lambda w: MatchedTimeTransformer(d_in, d_out, max(16, (w // 4) * 4), num_layers=num_layers)))
    w_s5 = _w(find_matched_width(target_params, lambda w: MatchedS5(d_in, d_out, w, num_layers=num_layers)))
    w_mamba = _w(find_matched_width(target_params, lambda w: MatchedMamba(d_in, d_out, w, num_layers=num_layers)))
    w_cde = _w(find_matched_width(target_params, lambda w: MatchedNeuralCDE(d_in, d_out, w, num_layers=num_layers)))

    return {
        "DAWIN_MLP_Parallel": ref_mlp,
        "DAWIN_MLP_Serial": ref_serial,
        "DAWIN_Transformer": dawin_tf,
        "ReLU_MLP": MatchedReLUMLP(d_in, d_out, w_relu, num_layers=num_layers),
        "Time_Transformer": MatchedTimeTransformer(d_in, d_out, max(16, (w_tf // 4) * 4), num_layers=num_layers),
        "S5": MatchedS5(d_in, d_out, w_s5, num_layers=num_layers),
        "Mamba": MatchedMamba(d_in, d_out, w_mamba, num_layers=num_layers),
        "Neural_CDE": MatchedNeuralCDE(d_in, d_out, w_cde, num_layers=num_layers),
    }


def run_b200_profile(
    results_dir: Path,
    precision: str = "bf16",
    device_str: Optional[str] = None,
    batch_sizes: Optional[List[int]] = None,
    seq_lengths: Optional[List[int]] = None,
    warmup_iters: int = 30,
    timed_iters: int = 200,
    profile_backward: bool = False,
) -> Dict[str, Any]:
    """
    Executes RQ4 GPU throughput and peak memory scaling benchmarks across batch sizes
    {1, 16, 256, 4096} and sequence lengths {32, 512, 4096, 16384}.
    """
    if batch_sizes is None:
        batch_sizes = [1, 16, 256, 4096]
    if seq_lengths is None:
        seq_lengths = [32, 512, 4096, 16384]

    use_cuda = torch.cuda.is_available() and (device_str is None or "cuda" in device_str)
    device = torch.device(device_str if device_str else ("cuda:0" if use_cuda else "cpu"))
    amp_dtype = configure_b200_runtime_policy(precision) if device.type == "cuda" else None

    d_in, d_out = 32, 16
    models = _build_b200_profile_models(d_in=d_in, d_out=d_out, ref_width=128, num_layers=2)
    records: List[Dict[str, Any]] = []

    print(f"\n[RQ4 B200 Profiler] Device={device} | Precision={precision} | Backward={profile_backward}")
    print(f"[RQ4 B200 Profiler] Batches={batch_sizes} | Lengths={seq_lengths}")

    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            # Guard against OOM on huge B*L combinations (> 131,072 transitions)
            if batch_size * seq_len > 131072 and device.type == "cuda":
                continue

            x = torch.randn(batch_size, seq_len, d_in, device=device)
            dt = torch.rand(batch_size, seq_len, device=device) * 0.5 + 0.01

            dawin_ref_tput = None
            dawin_ref_mem = None

            for name, model in models.items():
                # Skip quadratic attention or ODE solver on very long sequences to avoid hang/OOM
                if seq_len >= 4096 and name in ("DAWIN_Transformer", "Time_Transformer", "Neural_CDE"):
                    continue
                if seq_len >= 512 and batch_size >= 256 and name == "Neural_CDE":
                    continue

                model = model.to(device)
                model.train(profile_backward)
                params = count_trainable_parameters(model)

                try:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)

                    # Warmup iterations
                    for _ in range(warmup_iters):
                        if profile_backward:
                            model.zero_grad(set_to_none=True)
                            with torch.amp.autocast(
                                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
                            ):
                                out = model(x, dt)
                                loss = out.pow(2).mean()
                            loss.backward()
                        else:
                            with torch.no_grad():
                                with torch.amp.autocast(
                                    device_type="cuda",
                                    dtype=amp_dtype,
                                    enabled=amp_dtype is not None,
                                ):
                                    _ = model(x, dt)

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        start_ev = torch.cuda.Event(enable_timing=True)
                        end_ev = torch.cuda.Event(enable_timing=True)
                        start_ev.record()
                    else:
                        t0 = time.perf_counter()

                    for _ in range(timed_iters):
                        if profile_backward:
                            model.zero_grad(set_to_none=True)
                            with torch.amp.autocast(
                                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
                            ):
                                out = model(x, dt)
                                loss = out.pow(2).mean()
                            loss.backward()
                        else:
                            with torch.no_grad():
                                with torch.amp.autocast(
                                    device_type="cuda",
                                    dtype=amp_dtype,
                                    enabled=amp_dtype is not None,
                                ):
                                    _ = model(x, dt)

                    if device.type == "cuda":
                        end_ev.record()
                        torch.cuda.synchronize(device)
                        total_ms = start_ev.elapsed_time(end_ev)
                        peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                    else:
                        total_ms = (time.perf_counter() - t0) * 1000.0
                        peak_mib = 0.0

                    latency_ms = total_ms / timed_iters
                    transitions = batch_size * seq_len
                    tput_m = (transitions / (latency_ms / 1000.0)) / 1e6

                    if name == "DAWIN_MLP_Parallel":
                        dawin_ref_tput = tput_m
                        dawin_ref_mem = peak_mib

                    speedup_vs_comp = (dawin_ref_tput / tput_m) if (dawin_ref_tput and tput_m > 0) else 1.0
                    mem_saving_pct = (
                        100.0 * (1.0 - (dawin_ref_mem / peak_mib))
                        if (dawin_ref_mem is not None and peak_mib > 0)
                        else 0.0
                    )

                    rec = {
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "model": name,
                        "parameters": params,
                        "latency_ms": round(latency_ms, 4),
                        "throughput_m_trans_s": round(tput_m, 4),
                        "peak_memory_mib": round(peak_mib, 3),
                        "dawin_speedup_ratio": round(speedup_vs_comp, 3),
                        "dawin_mem_saving_pct": round(mem_saving_pct, 2),
                    }
                    records.append(rec)
                    print(
                        f"  L={seq_len:5d} | B={batch_size:4d} | {name:19s} | "
                        f"{latency_ms:7.3f} ms | {tput_m:7.3f} M/s | {peak_mib:7.2f} MiB"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  L={seq_len:5d} | B={batch_size:4d} | {name:19s} | OOM skipped")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise
                finally:
                    model.cpu()

    out_payload = {
        "precision": precision,
        "device": str(device),
        "profile_backward": profile_backward,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "records": records,
    }

    json_path = results_dir / f"rq4_b200_scaling_{precision}.json"
    csv_path = results_dir / f"rq4_b200_scaling_{precision}.csv"
    md_path = results_dir / f"rq4_b200_scaling_{precision}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    if records:
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        md_lines = [
            f"# RQ4 B200 GPU Throughput & Memory Scaling ({precision.upper()})",
            "",
            "| Seq Len (L) | Batch (B) | Model | Params | Latency (ms) | Throughput (10^6 trans/s) | Peak Mem (MiB) | DAWIN Speedup | DAWIN Mem Saving (%) |",
            "|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in records:
            md_lines.append(
                f"| {r['seq_len']} | {r['batch_size']} | {r['model']} | {r['parameters']:,} | "
                f"{r['latency_ms']:.3f} | {r['throughput_m_trans_s']:.3f} | {r['peak_memory_mib']:.2f} | "
                f"{r['dawin_speedup_ratio']:.2f}x | {r['dawin_mem_saving_pct']:.1f}% |"
            )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

    print(f"[RQ4 B200 Profiler] Saved JSON, CSV, and Markdown reports to {results_dir}")
    return out_payload


def consolidate_master_reports(
    results_dir: Path,
    precision: str,
    jobs: List[BenchmarkJob],
) -> Path:
    """Merges per-task JSON summaries into master_benchmark_summary_<precision>.json/.csv/.md."""
    master_tasks: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []

    for job in jobs:
        summary_file = results_dir / f"{job.backbone}_{job.task}_{precision}.json"
        if not summary_file.exists():
            continue
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        task_entry = {
            "suite": job.suite,
            "task": job.task,
            "backbone": job.backbone,
            "metric": job.metric_name,
            "higher_is_better": job.higher_is_better,
            "summary": data,
        }
        master_tasks.append(task_entry)

        for stats in data.get("summary", []):
            flat_rows.append(
                {
                    "suite": job.suite,
                    "task": job.task,
                    "backbone": job.backbone,
                    "metric": job.metric_name,
                    "model": stats.get("model", ""),
                    "parameters": stats.get("params", stats.get("parameters", 0)),
                    "val_mean": stats.get("val_mean", float("nan")),
                    "val_sd": stats.get("val_sd", float("nan")),
                    "test_mean": stats.get("test_mean", float("nan")),
                    "test_sd": stats.get("test_sd", float("nan")),
                    "delta_vs_dawin": stats.get("delta_vs_dawin", 0.0),
                    "ci_95": stats.get("ci_95", "—"),
                    "holm_p": stats.get("holm_p", "—"),
                    "dawin_wins": stats.get("dawin_wins", "—"),
                }
            )

    master_payload = {
        "precision": precision,
        "num_completed_tasks": len(master_tasks),
        "tasks": master_tasks,
    }

    master_json = results_dir / f"master_benchmark_summary_{precision}.json"
    master_csv = results_dir / f"master_benchmark_summary_{precision}.csv"
    master_md = results_dir / f"master_benchmark_summary_{precision}.md"

    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    if flat_rows:
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

        md_lines = [
            f"# DAWIN Master 10-Seed Benchmark Summary ({precision.upper()})",
            "",
            "| Suite | Task | Backbone | Metric | Model | Params | Test Mean ± SD | Δ vs DAWIN | 95% CI | Holm p | Wins |",
            "|:---|:---|:---|:---|:---|---:|---:|---:|:---|---:|:---:|",
        ]
        for r in flat_rows:
            md_lines.append(
                f"| {r['suite']} | {r['task']} | {r['backbone']} | {r['metric']} | {r['model']} | "
                f"{r['parameters']:,} | {r['test_mean']:.4f} ± {r['test_sd']:.4f} | "
                f"{r['delta_vs_dawin']:+.4f} | {r['ci_95']} | {r['holm_p']} | {r['dawin_wins']} |"
            )
        with open(master_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

    print(f"[Consolidator] Wrote {master_json} ({len(master_tasks)} tasks)")
    return master_json


def export_latex_tables_automatically(
    results_dir: Path,
    precision: str,
    venue: str = "neurips",
    tex_output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Automatically invokes `export_dawin_latex_tables.py` at the end of batch runs
    so publication-ready LaTeX `table*` blocks generate immediately.
    """
    script_dir = Path(__file__).resolve().parent
    exporter_script = script_dir / "export_dawin_latex_tables.py"
    if not exporter_script.exists():
        print(f"[LaTeX Exporter] Warning: {exporter_script} not found; skipping automatic LaTeX export.")
        return None

    out_tex = tex_output_path or (results_dir / f"dawin_{venue}_tables_{precision}.tex")
    cmd = [
        sys.executable,
        str(exporter_script),
        "--results-dir",
        str(results_dir),
        "--precision",
        precision,
        "--venue",
        venue,
        "--output",
        str(out_tex),
    ]
    print(f"\n[LaTeX Exporter] Generating {venue.upper()} LaTeX tables -> {out_tex}")
    res = subprocess.run(cmd, check=False)
    if res.returncode == 0:
        print(f"[LaTeX Exporter] Successfully wrote {out_tex}")
        return out_tex
    print(f"[LaTeX Exporter] Failed with exit code {res.returncode}")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Multi-Task Batch Runner, B200 Profiler & LaTeX Exporter for DAWIN"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["synthetic", "mujoco", "physionet", "trajectories"],
        choices=["synthetic", "mujoco", "physionet", "trajectories"],
        help="Benchmark suites to execute.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional explicit list of task names to run (overrides --suites).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Training and profiling precision policy (default: bf16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target PyTorch device (e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(101, 111)),
        help="Random seeds for final training runs (default: 101..110).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override default max epochs per task.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size per task.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_dawin",
        help="Directory for JSON, CSV, Markdown, and LaTeX outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks even if summary JSON already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining tasks if a single task fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    parser.add_argument(
        "--profile-b200",
        action="store_true",
        help="Run RQ4 B200 GPU throughput & memory scaling profiler before training sweeps.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Run only RQ4 B200 GPU profiling and LaTeX export (skip 10-seed training sweeps).",
    )
    parser.add_argument(
        "--profile-backward",
        action="store_true",
        help="Include backward pass timing in RQ4 B200 profiling.",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default="neurips",
        choices=["neurips", "iclr", "icml"],
        help="Target conference style for automated LaTeX table export (default: neurips).",
    )
    parser.add_argument(
        "--skip-latex-export",
        action="store_true",
        help="Disable automatic invocation of export_dawin_latex_tables.py at the end of the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    runner_script = script_dir / "run_dawin_experiments.py"

    if args.tasks:
        selected_jobs = [j for j in BENCHMARK_MANIFEST if j.task in args.tasks]
    else:
        selected_jobs = [j for j in BENCHMARK_MANIFEST if j.suite in args.suites]

    # 1. Optional RQ4 B200 Scaling Profile
    if args.profile_b200 or args.profile_only:
        if args.dry_run:
            print(f"[DRY-RUN] Would run RQ4 B200 profiler (precision={args.precision}, backward={args.profile_backward})")
        else:
            run_b200_profile(
                results_dir=results_dir,
                precision=args.precision,
                device_str=args.device,
                profile_backward=args.profile_backward,
            )

    # 2. Sequential 10-Seed Training Sweeps
    if not args.profile_only:
        print(f"\n[Batch Runner] Executing {len(selected_jobs)} benchmark tasks | Precision={args.precision}")
        for idx, job in enumerate(selected_jobs, start=1):
            summary_path = results_dir / f"{job.backbone}_{job.task}_{args.precision}.json"
            if summary_path.exists() and not args.force:
                print(f"[{idx}/{len(selected_jobs)}] Skipping {job.task} (already exists: {summary_path})")
                continue

            epochs = args.epochs if args.epochs is not None else job.default_epochs
            batch_size = args.batch_size if args.batch_size is not None else job.default_batch_size

            cmd = [
                sys.executable,
                str(runner_script),
                "--backbone",
                job.backbone,
                "--task",
                job.task,
                "--precision",
                args.precision,
                "--epochs",
                str(epochs),
                "--hidden-dim",
                str(job.default_width),
                "--num-layers",
                str(job.default_layers),
                "--out-dir",
                str(results_dir),
                "--seeds",
                *[str(s) for s in args.seeds],
            ]

            print(f"\n[{idx}/{len(selected_jobs)}] Running {job.suite}/{job.task} ({job.backbone})...")
            if args.dry_run:
                print("  CMD:", " ".join(cmd))
                continue

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[Batch Runner] ERROR on {job.task}: exit code {exc.returncode}")
                if not args.continue_on_error:
                    sys.exit(exc.returncode)

        if not args.dry_run:
            consolidate_master_reports(results_dir, args.precision, selected_jobs)

    # 3. Automated LaTeX Table Export
    if not args.skip_latex_export:
        if args.dry_run:
            print(
                f"[DRY-RUN] Would invoke export_dawin_latex_tables.py "
                f"--results-dir {results_dir} --precision {args.precision} --venue {args.venue}"
            )
        else:
            export_latex_tables_automatically(
                results_dir=results_dir,
                precision=args.precision,
                venue=args.venue,
            )


if __name__ == "__main__":
    main()
