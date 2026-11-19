#!/usr/bin/env python3
"""
plot_dawin_figures.py

Generates publication-ready PDF and PNG figures for the DAWIN manuscript
directly from master_benchmark_summary_<precision>.json and
rq4_b200_scaling_<precision>.json.

Figures generated:
  1. fig_rq1_main_comparison.<ext>: Multi-panel parameter-matched accuracy
     comparison across tasks (10-seed Mean +/- SD, horizontal dot-whisker plot).
  2. fig_rq3_dt_interventions.<ext>: Paired timing-intervention diagnostics
     showing relative (%) and absolute error changes under batch-shuffled
     (dt_perm) and constant-mean (dt_const) intervals with 95% bootstrap CIs,
     highlighting invariance on the dt-irrelevant negative control.
  3. fig_rq4_b200_scaling.<ext>: Two-panel NVIDIA B200 GPU scaling plot
     (log-log per-batch latency / throughput vs. batch size B in {1, 16, 256, 4096},
     and log-log peak allocated GPU memory in MiB) contrasting DAWIN (Parallel Scan)
     against DAWIN (Serial), Transformer, S5, Mamba, and Neural CDE.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Okabe-Ito colorblind-safe palette & consistent styling
MODEL_STYLES: Dict[str, Dict[str, Any]] = {
    "dawin": {
        "label": "DAWIN (Ours)",
        "color": "#B22222",
        "marker": "o",
        "linestyle": "-",
        "zorder": 10,
    },
    "DAWIN_MLP_Parallel": {
        "label": "DAWIN-MLP (Parallel Scan)",
        "color": "#B22222",
        "marker": "o",
        "linestyle": "-",
        "zorder": 10,
    },
    "DAWIN_MLP_Serial": {
        "label": "DAWIN-MLP (Serial)",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
        "zorder": 8,
    },
    "DAWIN_Transformer": {
        "label": "DAWIN-Transformer",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "-",
        "zorder": 9,
    },
    "s5": {
        "label": "S5",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
        "zorder": 6,
    },
    "S5": {
        "label": "S5",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
        "zorder": 6,
    },
    "mamba": {
        "label": "Mamba",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
        "zorder": 6,
    },
    "Mamba": {
        "label": "Mamba",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
        "zorder": 6,
    },
    "neural_cde": {
        "label": "Neural CDE",
        "color": "#E69F00",
        "marker": "v",
        "linestyle": "-.",
        "zorder": 5,
    },
    "Neural_CDE": {
        "label": "Neural CDE",
        "color": "#E69F00",
        "marker": "v",
        "linestyle": "-.",
        "zorder": 5,
    },
    "time_transformer": {
        "label": "Time-Transformer",
        "color": "#1E293B",
        "marker": "^",
        "linestyle": "-",
        "zorder": 5,
    },
    "Time_Transformer": {
        "label": "Time-Transformer",
        "color": "#1E293B",
        "marker": "^",
        "linestyle": "-",
        "zorder": 5,
    },
    "relu_mlp": {
        "label": "ReLU-MLP",
        "color": "#64748B",
        "marker": "d",
        "linestyle": ":",
        "zorder": 4,
    },
    "ReLU_MLP": {
        "label": "ReLU-MLP",
        "color": "#64748B",
        "marker": "d",
        "linestyle": ":",
        "zorder": 4,
    },
    "gru_gnn": {
        "label": "GRU-GNN",
        "color": "#56B4E9",
        "marker": "P",
        "linestyle": "--",
        "zorder": 4,
    },
}

TASK_DISPLAY_NAMES: Dict[str, str] = {
    "damped_oscillator": "Damped Oscillator",
    "dt_irrelevant": r"$\Delta t$-Irrelevant Control",
    "mujoco_halfcheetah": "HalfCheetah-v5",
    "mujoco_hopper": "Hopper-v5 (L=4)",
    "mujoco_reacher": "Reacher-v5",
    "mujoco_walker2d": "Walker2d-v5",
    "physionet_mortality": "PhysioNet-12 (Mortality)",
    "physionet_imputation": "PhysioNet-12 (Imputation)",
    "physionet_dense_wsp": "PhysioNet-12 (Dense)",
    "eth_ucy_jaad_pos": "ETH/UCY+JAAD (Pos-Only)",
    "eth_ucy_jaad_motion": "ETH/UCY+JAAD (Motion)",
}


def apply_publication_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def get_style(model_key: str) -> Dict[str, Any]:
    if model_key in MODEL_STYLES:
        return MODEL_STYLES[model_key]
    return {
        "label": model_key.replace("_", " "),
        "color": "#475569",
        "marker": "o",
        "linestyle": "-",
        "zorder": 3,
    }


def pretty_task_name(task_key: str) -> str:
    return TASK_DISPLAY_NAMES.get(task_key, task_key.replace("_", " ").title())


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: List[str], dpi: int = 300) -> List[Path]:
    saved = []
    for ext in formats:
        ext_clean = ext.lstrip(".")
        out_path = out_dir / f"{stem}.{ext_clean}"
        fig.savefig(out_path, format=ext_clean, dpi=dpi)
        saved.append(out_path)
    plt.close(fig)
    return saved


# =============================================================================
# Figure 1: Main Parameter-Matched Accuracy Comparison (RQ1 & RQ2)
# =============================================================================

def plot_rq1_main_comparison(
    master_data: Dict[str, Any],
    out_dir: Path,
    formats: List[str],
    dpi: int = 300,
) -> List[Path]:
    tasks_dict: Dict[str, Any] = master_data.get("tasks", {})
    if not tasks_dict:
        return []

    task_keys = list(tasks_dict.keys())
    n_tasks = len(task_keys)
    ncols = min(3, n_tasks)
    nrows = math.ceil(n_tasks / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.5 * ncols, 2.35 * nrows),
        squeeze=False,
    )

    for idx, task_key in enumerate(task_keys):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        task_entry = tasks_dict[task_key]
        models_entry = task_entry.get("models", {})
        metric_name = task_entry.get("metric_name", "Test NRMSE")
        higher_is_better = bool(task_entry.get("higher_is_better", False))

        # Order models: DAWIN at top, then comparators
        ordered_keys = []
        if "dawin" in models_entry:
            ordered_keys.append("dawin")
        for k in ["s5", "mamba", "neural_cde", "time_transformer", "relu_mlp", "gru_gnn"]:
            if k in models_entry and k not in ordered_keys:
                ordered_keys.append(k)
        for k in models_entry:
            if k not in ordered_keys:
                ordered_keys.append(k)

        y_positions = np.arange(len(ordered_keys))[::-1]
        for y_pos, m_key in zip(y_positions, ordered_keys):
            m_stats = models_entry[m_key]
            mean_val = float(m_stats.get("test_mean", m_stats.get("mean", 0.0)))
            std_val = float(m_stats.get("test_std", m_stats.get("std", 0.0)))
            style = get_style(m_key)

            ax.errorbar(
                mean_val,
                y_pos,
                xerr=std_val,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.2,
                capsize=2.5,
                markersize=5.5 if m_key == "dawin" else 4.8,
                zorder=style["zorder"],
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([get_style(k)["label"] for k in ordered_keys])
        ax.grid(True, axis="x", alpha=0.8)
        direction_hint = "higher is better" if higher_is_better else "lower is better"
        ax.set_title(f"({chr(97 + idx)}) {pretty_task_name(task_key)}", fontweight="bold", loc="left")
        ax.set_xlabel(f"{metric_name} ({direction_hint})", fontsize=8.0)

        if "irrelevant" in task_key:
            ax.set_xscale("log")

    # Hide unused subplots
    for idx in range(n_tasks, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.tight_layout(w_pad=1.4, h_pad=1.3)
    return save_figure(fig, out_dir, "fig_rq1_main_comparison", formats, dpi=dpi)


# =============================================================================
# Figure 2: RQ3 Interval-Shuffling & Constant-Interval Diagnostics
# =============================================================================

def plot_rq3_dt_interventions(
    master_data: Dict[str, Any],
    out_dir: Path,
    formats: List[str],
    dpi: int = 300,
) -> List[Path]:
    tasks_dict: Dict[str, Any] = master_data.get("tasks", {})
    if not tasks_dict:
        return []

    rows: List[Dict[str, Any]] = []
    for task_key, task_entry in tasks_dict.items():
        dawin_entry = task_entry.get("models", {}).get("dawin", {})
        interv = dawin_entry.get("dt_interventions", task_entry.get("dt_interventions", {}))
        if not interv:
            continue

        base_mean = float(dawin_entry.get("test_mean", dawin_entry.get("mean", 1.0)))
        higher_is_better = bool(task_entry.get("higher_is_better", False))
        sign = -1.0 if higher_is_better else 1.0  # Positive = degradation under intervention

        shuf = interv.get("shuffled", interv.get("dt_perm", {}))
        const = interv.get("constant", interv.get("dt_const", {}))

        if not shuf and not const:
            continue

        def extract_pct(d: Dict[str, Any]) -> Tuple[float, float, float]:
            if "rel_change_pct" in d:
                val = sign * float(d["rel_change_pct"])
                ci = d.get("rel_ci_95", [val, val])
                lo, hi = sorted([sign * float(ci[0]), sign * float(ci[1])])
                return val, val - lo, hi - val
            delta = sign * float(d.get("delta_mean", d.get("effect", 0.0)))
            ci = d.get("ci_95", [delta, delta])
            denom = max(abs(base_mean), 1e-8)
            pct = 100.0 * delta / denom
            lo, hi = sorted([100.0 * sign * float(ci[0]) / denom, 100.0 * sign * float(ci[1]) / denom])
            return pct, pct - lo, hi - pct

        shuf_pct, shuf_err_lo, shuf_err_hi = extract_pct(shuf) if shuf else (0.0, 0.0, 0.0)
        const_pct, const_err_lo, const_err_hi = extract_pct(const) if const else (0.0, 0.0, 0.0)

        rows.append({
            "task_key": task_key,
            "label": pretty_task_name(task_key),
            "is_control": "irrelevant" in task_key,
            "shuf_pct": shuf_pct,
            "shuf_err": [[shuf_err_lo], [shuf_err_hi]],
            "const_pct": const_pct,
            "const_err": [[const_err_lo], [const_err_hi]],
        })

    if not rows:
        return []

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(6.6, max(2.4, 0.48 * n_rows + 0.9)))
    y_base = np.arange(n_rows)[::-1]
    offset = 0.16

    ax.axvline(0.0, color="#0F172A", linewidth=1.0, zorder=2)

    for idx, (y_pos, row) in enumerate(zip(y_base, rows)):
        if row["is_control"]:
            ax.axhspan(y_pos - 0.45, y_pos + 0.45, color="#F1F5F9", zorder=0)

        # Shuffled intervals (dt_perm)
        ax.errorbar(
            row["shuf_pct"],
            y_pos + offset,
            xerr=row["shuf_err"],
            fmt="o",
            color="#B22222",
            ecolor="#B22222",
            elinewidth=1.3,
            capsize=2.8,
            markersize=5.5,
            label=r"Shuffled intervals ($\Delta t_{\mathrm{perm}}$)" if idx == 0 else None,
            zorder=5,
        )
        # Constant mean intervals (dt_const)
        ax.errorbar(
            row["const_pct"],
            y_pos - offset,
            xerr=row["const_err"],
            fmt="s",
            color="#0072B2",
            ecolor="#0072B2",
            elinewidth=1.3,
            capsize=2.8,
            markersize=5.0,
            label=r"Constant mean intervals ($\Delta t_{\mathrm{const}}$)" if idx == 0 else None,
            zorder=5,
        )

        # Inline percentage annotation for shuffled intervention
        right_edge = row["shuf_pct"] + row["shuf_err"][1][0]
        left_edge = row["shuf_pct"] - row["shuf_err"][0][0]
        if row["shuf_pct"] >= 0:
            ax.text(
                right_edge + 0.35,
                y_pos + offset,
                f"{row['shuf_pct']:+.2f}%",
                va="center",
                ha="left",
                fontsize=8.0,
                color="#7F1D1D",
                fontweight="bold" if not row["is_control"] else "normal",
            )
        else:
            ax.text(
                left_edge - 0.35,
                y_pos + offset,
                f"{row['shuf_pct']:+.2f}%",
                va="center",
                ha="right",
                fontsize=8.0,
                color="#334155",
            )

    ax.set_yticks(y_base)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel(r"Relative degradation in DAWIN test error under $\Delta t$ intervention (%)")
    ax.set_title(
        "RQ3: Causal Utilization of Physical Intervals (Paired Seed-Wise Change, 95% Bootstrap CI)",
        fontweight="bold",
        loc="left",
    )
    ax.grid(True, axis="x", alpha=0.8)

    # Pad x-limits so annotations never clip
    xmin, xmax = ax.get_xlim()
    span = max(xmax - xmin, 1.0)
    ax.set_xlim(xmin - 0.14 * span, xmax + 0.18 * span)

    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CBD5E1")
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_rq3_dt_interventions", formats, dpi=dpi)


# =============================================================================
# Figure 3: RQ4 NVIDIA B200 Throughput & Peak Memory Scaling
# =============================================================================

def plot_rq4_b200_scaling(
    rq4_data: Dict[str, Any],
    out_dir: Path,
    formats: List[str],
    target_seq_len: Optional[int] = None,
    dpi: int = 300,
) -> List[Path]:
    records = rq4_data.get("records", rq4_data.get("results", []))
    if not records:
        return []

    # Group records by sequence length L and choose target L (default: largest available or 512/32)
    seq_lens = sorted({int(r.get("seq_len", r.get("L", 32))) for r in records})
    if not seq_lens:
        return []
    if target_seq_len is not None and target_seq_len in seq_lens:
        chosen_L = target_seq_len
    elif 512 in seq_lens:
        chosen_L = 512
    else:
        chosen_L = seq_lens[-1]

    filtered = [r for r in records if int(r.get("seq_len", r.get("L", 32))) == chosen_L]
    if not filtered:
        return []

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in filtered:
        m_name = str(r.get("model", r.get("model_name", "unknown")))
        by_model.setdefault(m_name, []).append(r)

    preferred_order = [
        "DAWIN_MLP_Parallel",
        "DAWIN_Transformer",
        "DAWIN_MLP_Serial",
        "Time_Transformer",
        "S5",
        "Mamba",
        "Neural_CDE",
        "ReLU_MLP",
    ]
    model_keys = [m for m in preferred_order if m in by_model] + [
        m for m in by_model if m not in preferred_order
    ]

    fig, (ax_lat, ax_mem) = plt.subplots(1, 2, figsize=(7.4, 3.1))

    for m_key in model_keys:
        rows = sorted(by_model[m_key], key=lambda item: int(item.get("batch_size", item.get("B", 1))))
        batches = np.array([int(r.get("batch_size", r.get("B", 1))) for r in rows], dtype=float)
        lat_mean = np.array([float(r.get("latency_ms_mean", r.get("latency_ms", 0.0))) for r in rows], dtype=float)
        lat_std = np.array([float(r.get("latency_ms_std", 0.0)) for r in rows], dtype=float)
        mem_mib = np.array([float(r.get("peak_memory_mib", r.get("peak_mib", 0.0))) for r in rows], dtype=float)

        style = get_style(m_key)
        plot_kwargs = dict(
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.7 if "DAWIN" in m_key else 1.3,
            markersize=5.2,
            zorder=style["zorder"],
        )

        # Panel (a): Per-batch latency (ms)
        ax_lat.plot(batches, lat_mean, **plot_kwargs)
        if np.any(lat_std > 0):
            ax_lat.fill_between(
                batches,
                np.maximum(lat_mean - lat_std, 1e-4),
                lat_mean + lat_std,
                color=style["color"],
                alpha=0.14,
                zorder=style["zorder"] - 1,
            )

        # Panel (b): Peak allocated GPU memory (MiB)
        ax_mem.plot(batches, mem_mib, **plot_kwargs)

    batch_ticks = sorted({int(r.get("batch_size", r.get("B", 1))) for r in filtered})
    for ax, title, ylabel in [
        (ax_lat, f"(a) Per-Batch Latency at L = {chosen_L} (Lower is Better)", "Per-batch latency (ms)"),
        (ax_mem, f"(b) Peak Allocated Memory at L = {chosen_L} (Lower is Better)", "Peak allocated GPU memory (MiB)"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batch_ticks)
        ax.set_xticklabels([str(b) for b in batch_ticks])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Batch size (B)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", loc="left", fontsize=9.2)
        ax.grid(True, which="major", alpha=0.85)

    handles, labels = ax_lat.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=min(4, len(labels)),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
    )
    fig.tight_layout(w_pad=1.6)
    return save_figure(fig, out_dir, "fig_rq4_b200_scaling", formats, dpi=dpi)


# =============================================================================
# CLI Entrypoint
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render publication-ready DAWIN figures from JSON benchmark summaries."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_dawin",
        help="Directory containing master_benchmark_summary_<prec>.json and rq4_b200_scaling_<prec>.json",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Precision tag of the JSON artifacts to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write generated figures (defaults to <results-dir>/figures).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output image formats (e.g., pdf png eps).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Specific sequence length L to plot for RQ4 scaling (defaults to 512 or max available).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for raster formats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_publication_rcparams()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    master_path = results_dir / f"master_benchmark_summary_{args.precision}.json"
    rq4_path = results_dir / f"rq4_b200_scaling_{args.precision}.json"

    master_data = load_json_if_exists(master_path)
    rq4_data = load_json_if_exists(rq4_path)

    generated: List[Path] = []
    if master_data is not None:
        generated.extend(plot_rq1_main_comparison(master_data, out_dir, args.formats, dpi=args.dpi))
        generated.extend(plot_rq3_dt_interventions(master_data, out_dir, args.formats, dpi=args.dpi))
    else:
        print(f"[warn] Master benchmark summary not found at {master_path}; skipping RQ1 and RQ3 plots.")

    if rq4_data is not None:
        generated.extend(
            plot_rq4_b200_scaling(rq4_data, out_dir, args.formats, target_seq_len=args.seq_len, dpi=args.dpi)
        )
    else:
        print(f"[warn] RQ4 B200 scaling JSON not found at {rq4_path}; skipping RQ4 plot.")

    if generated:
        print(f"Generated {len(generated)} figure file(s) in {out_dir}:")
        for p in generated:
            print(f"  - {p}")
    else:
        print("No figures were generated. Check that JSON files exist and contain valid records.")


if __name__ == "__main__":
    main()
