#!/usr/bin/env python3
"""
export_dawin_latex_tables.py
----------------------------
Publication-ready LaTeX table exporter for DAWIN benchmark and GPU scaling JSON
artifacts (master_benchmark_summary_<precision>.json and rq4_b200_scaling_<precision>.json).

Generates clean, booktabs-formatted NeurIPS / ICLR / ICML \begin{table*} blocks:
  1. Table 1 (Main Accuracy & Statistical Comparison): Mean +/- SD across 10 seeds,
     bolding best per-task results, underlining runner-up, and annotating Holm-Bonferroni
     adjusted p-values (* p < 0.05) and seed win counts.
  2. Table 2 (RQ3 Causal Timing Interventions): Paired Delta-NRMSE under batch-shuffled
     (Delta t_perm) and constant-mean (Delta t_const) intervals with 95% bootstrap CIs.
  3. Table 3 (RQ4 NVIDIA B200 Throughput & Memory Scaling): Batch sizes B in {1, 16, 256, 4096},
     latency (ms), throughput (10^6 transitions/s), peak allocated memory (MiB),
     parallel scan speedup ratio, and memory savings (%).

Usage:
    python export_dawin_latex_tables.py \
        --results-dir results_dawin \
        --precision bf16 \
        --venue neurips \
        --output generated/dawin_paper_tables.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "dawin": r"\textbf{DAWIN (Ours)}",
    "dawin_mlp": r"\textbf{DAWIN-MLP (Ours)}",
    "dawin_transformer": r"\textbf{DAWIN-Transformer (Ours)}",
    "dawin_gnn": r"\textbf{DAWIN-GNN (Ours)}",
    "DAWIN_MLP_Parallel": r"\textbf{DAWIN-MLP (Triton Scan)}",
    "DAWIN_MLP_Serial": r"DAWIN-MLP (Serial Loop)",
    "DAWIN_Transformer": r"\textbf{DAWIN-Transformer}",
    "relu_mlp": r"ReLU-MLP (Matched)",
    "ReLU_MLP": r"ReLU-MLP",
    "time_transformer": r"Time-Transformer",
    "Time_Transformer": r"Time-Transformer",
    "neural_cde": r"Neural CDE",
    "Neural_CDE": r"Neural CDE",
    "s5": r"S5 (ZOH)",
    "S5": r"S5 (ZOH)",
    "mamba": r"Mamba (Selective)",
    "Mamba": r"Mamba (Selective)",
    "gru_gnn": r"GRU-GNN (Matched)",
    "wfb": r"WFB (Stateless)",
    "wsp": r"WSP (Carry)",
    "contiformer": r"ContiFormer",
}

TASK_DISPLAY_NAMES: Dict[str, str] = {
    "damped_oscillator": r"Damped Oscillator ($L=32$)",
    "dt_irrelevant": r"$\Delta t$-Irrelevant Control",
    "mujoco_halfcheetah": r"HalfCheetah-v5",
    "mujoco_hopper": r"Hopper-v5 ($L=4$)",
    "mujoco_reacher": r"Reacher-v5",
    "mujoco_walker2d": r"Walker2d-v5",
    "physionet_mortality": r"PhysioNet-12 (Mortality)",
    "physionet_imputation": r"PhysioNet-12 (Imputation)",
    "physionet_dense_wsp": r"PhysioNet-12 (Dense)",
    "eth_ucy_jaad_pos": r"ETH/UCY+JAAD (Pos-Only)",
    "eth_ucy_jaad_motion": r"ETH/UCY+JAAD (Motion)",
}

# Tasks where higher metric is better (e.g., AUROC / AUPRC / MRR)
HIGHER_IS_BETTER_TASKS = {"physionet_mortality", "wikipedia_link", "reddit_link"}


def _fmt_float(val: Optional[float], decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return r"--"
    if abs(val) > 0 and abs(val) < 10 ** (-decimals):
        return f"{val:.2e}"
    return f"{val:.{decimals}f}"


def _fmt_mean_sd(mean: Optional[float], sd: Optional[float], decimals: int = 4, bold: bool = False, underline: bool = False) -> str:
    if mean is None or math.isnan(mean):
        return r"--"
    sd_val = 0.0 if (sd is None or math.isnan(sd)) else sd
    core = f"{mean:.{decimals}f} \\pm {sd_val:.{decimals}f}"
    if bold:
        return f"$\\mathbf{{{core}}}$"
    if underline:
        return f"$\\underline{{{core}}}$"
    return f"${core}$"


def _fmt_ci(low: Optional[float], high: Optional[float], decimals: int = 4) -> str:
    if low is None or high is None or math.isnan(low) or math.isnan(high):
        return r"--"
    return f"$[{low:.{decimals}f},\, {high:.{decimals}f}]$"


def _fmt_p_value(p: Optional[float]) -> str:
    if p is None or math.isnan(p):
        return r"--"
    if p < 0.001:
        return r"$<0.001^*$"
    if p < 0.05:
        return f"${p:.3f}^*$"
    return f"${p:.3f}$"


def _venue_header_comment(venue: str) -> str:
    venue_norm = venue.lower().strip()
    if venue_norm == "neurips":
        return (
            "% Formatted for NeurIPS (neurips_2025.sty / neurips_2026.sty)\n"
            "% Required preamble packages: \usepackage{booktabs,multirow,amsmath,amssymb,graphicx}"
        )
    if venue_norm == "iclr":
        return (
            "% Formatted for ICLR (iclr2026_conference.sty)\n"
            "% Required preamble packages: \usepackage{booktabs,multirow,amsmath,amssymb,graphicx}"
        )
    return (
        "% Formatted for ICML (icml2026.sty)\n"
        "% Required preamble packages: \usepackage{booktabs,multirow,amsmath,amssymb,graphicx}"
    )


def _extract_task_records(master_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalizes master_benchmark_summary_<precision>.json across schema versions."""
    if isinstance(master_data, list):
        return master_data
    if "tasks" in master_data and isinstance(master_data["tasks"], list):
        return master_data["tasks"]
    if "results" in master_data and isinstance(master_data["results"], dict):
        records = []
        for task_key, payload in master_data["results"].items():
            rec = dict(payload)
            rec.setdefault("task", task_key)
            records.append(rec)
        return records
    # Fallback: top-level keys are task names
    records = []
    for k, v in master_data.items():
        if isinstance(v, dict) and ("models" in v or "comparisons" in v or "dawin" in v):
            rec = dict(v)
            rec.setdefault("task", k)
            records.append(rec)
    return records


def generate_main_benchmark_table(
    master_data: Dict[str, Any],
    venue: str = "neurips",
    decimals: int = 4,
) -> str:
    """
    Generates Table 1: Cross-benchmark 10-seed parameter-matched accuracy & paired statistical
    comparison table (\begin{table*}).
    """
    records = _extract_task_records(master_data)
    lines: List[str] = [
        _venue_header_comment(venue),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Main Parameter-Matched Benchmark Comparison (10 Final Seeds, $\pm$ SD).} "
        r"All models share identical train/validation/test splits, early-stopping criteria, and parameter budgets within $\pm 2\%$. "
        r"\textbf{Bold} denotes best mean performance per task; \underline{underline} denotes runner-up. "
        r"$\Delta$ reports paired comparator-minus-DAWIN error ($\Delta > 0$ favors DAWIN on error metrics; inverted on AUROC). "
        r"$^*$ indicates $p < 0.05$ after Holm--Bonferroni correction across exact two-sided sign-flip tests.}",
        r"\label{tab:dawin_main_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Benchmark Task} & \textbf{Model} & \textbf{Width} & \textbf{Params} & \textbf{Val. Metric} & \textbf{Test Metric} & \textbf{Paired $\Delta$ (95\% CI)} & \textbf{Holm $p$} & \textbf{Wins} \\",
        r"\midrule",
    ]

    if not records:
        lines.append(r"\multicolumn{9}{c}{\textit{No completed benchmark records found in master summary JSON.}} \\")
    else:
        for idx, rec in enumerate(records):
            task_id = rec.get("task", f"task_{idx}")
            task_label = TASK_DISPLAY_NAMES.get(task_id, task_id.replace("_", r"\_"))
            higher_better = rec.get("higher_is_better", task_id in HIGHER_IS_BETTER_TASKS)
            models_dict: Dict[str, Any] = rec.get("models", {})
            comps_dict: Dict[str, Any] = rec.get("comparisons", {})

            # Order models: DAWIN first, then canonical comparators
            ordered_keys = ["dawin", "relu_mlp", "time_transformer", "neural_cde", "s5", "mamba", "gru_gnn"]
            present_keys = [k for k in ordered_keys if k in models_dict] + [
                k for k in models_dict.keys() if k not in ordered_keys
            ]

            # Identify best and second-best test means
            valid_scores: List[Tuple[str, float]] = []
            for k in present_keys:
                m_info = models_dict[k]
                t_mean = m_info.get("test_mean", m_info.get("test_nrmse_mean"))
                if t_mean is not None and not math.isnan(t_mean):
                    valid_scores.append((k, float(t_mean)))

            valid_scores.sort(key=lambda pair: pair[1], reverse=higher_better)
            best_key = valid_scores[0][0] if len(valid_scores) >= 1 else None
            second_key = valid_scores[1][0] if len(valid_scores) >= 2 else None

            n_rows = max(1, len(present_keys))
            for r_i, m_key in enumerate(present_keys):
                m_info = models_dict[m_key]
                c_info = comps_dict.get(m_key, {})

                task_col = f"\multirow{{{n_rows}}}{{*}}{{{task_label}}}" if r_i == 0 else ""
                m_label = MODEL_DISPLAY_NAMES.get(m_key, m_key.replace("_", r"\_"))
                width_str = str(m_info.get("width", r"--"))
                params_val = m_info.get("params", m_info.get("num_params"))
                params_str = f"{int(params_val):,}" if params_val is not None else r"--"

                v_mean = m_info.get("val_mean", m_info.get("val_nrmse_mean"))
                v_sd = m_info.get("val_sd", m_info.get("val_nrmse_sd"))
                t_mean = m_info.get("test_mean", m_info.get("test_nrmse_mean"))
                t_sd = m_info.get("test_sd", m_info.get("test_nrmse_sd"))

                val_str = _fmt_mean_sd(v_mean, v_sd, decimals=decimals)
                test_str = _fmt_mean_sd(
                    t_mean,
                    t_sd,
                    decimals=decimals,
                    bold=(m_key == best_key),
                    underline=(m_key == second_key),
                )

                if m_key == "dawin":
                    delta_str = r"\textit{Reference}"
                    p_str = r"--"
                    wins_str = r"--"
                else:
                    d_mean = c_info.get("delta_mean", c_info.get("mean_diff"))
                    ci_lo = c_info.get("ci_low", c_info.get("bootstrap_ci_low"))
                    ci_hi = c_info.get("ci_high", c_info.get("bootstrap_ci_high"))
                    if d_mean is not None:
                        delta_str = f"${d_mean:+.{decimals}f}$ {_fmt_ci(ci_lo, ci_hi, decimals=decimals)}"
                    else:
                        delta_str = r"--"
                    p_str = _fmt_p_value(c_info.get("holm_p", c_info.get("p_holm")))
                    wins = c_info.get("dawin_wins")
                    total_seeds = c_info.get("n_seeds", 10)
                    wins_str = f"{wins}/{total_seeds}" if wins is not None else r"--"

                lines.append(
                    f"{task_col} & {m_label} & {width_str} & {params_str} & "
                    f"{val_str} & {test_str} & {delta_str} & {p_str} & {wins_str} \\"
                )

            if idx < len(records) - 1:
                lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_timing_intervention_table(
    master_data: Dict[str, Any],
    venue: str = "neurips",
    decimals: int = 4,
) -> str:
    """
    Generates Table 2 (RQ3): Causal Elapsed-Time Intervention Diagnostics
    (Batch-Shuffled Delta t_perm and Constant-Mean Delta t_const).
    """
    records = _extract_task_records(master_data)
    lines: List[str] = [
        _venue_header_comment(venue),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{RQ3 Causal Timing-Intervention Diagnostics on Trained DAWIN Models (10 Final Seeds).} "
        r"Holding trained parameters, spatial inputs $\mathbf{x}_i$, and ground-truth targets fixed at test time, "
        r"we report paired error degradation ($\Delta_{\text{int}} = \text{Error}_{\text{intervened}} - \text{Error}_{\text{true}}$) "
        r"under batch-shuffled intervals ($\Delta t_{\text{perm}}$) and constant batch-mean intervals ($\Delta t_{\text{const}}$). "
        r"Positive shifts confirm causal reliance on physical elapsed time; near-zero shift on the $\Delta t$-irrelevant control confirms absence of spurious timing bias.}",
        r"\label{tab:dawin_rq3_interventions}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Benchmark Task} & \textbf{Shuffled $\Delta t_{\text{perm}}$ Effect ($\pm$ SD)} & \textbf{95\% Bootstrap CI} & \textbf{Constant $\Delta t_{\text{const}}$ Effect ($\pm$ SD)} & \textbf{95\% Bootstrap CI} \\",
        r"\midrule",
    ]

    has_any = False
    for rec in records:
        task_id = rec.get("task", "unknown")
        task_label = TASK_DISPLAY_NAMES.get(task_id, task_id.replace("_", r"\_"))
        interventions = rec.get("dt_interventions", rec.get("timing_interventions", {}))
        if not interventions:
            continue
        has_any = True
        shuf = interventions.get("shuffled", {})
        const = interventions.get("constant", {})

        shuf_str = _fmt_mean_sd(shuf.get("mean_diff"), shuf.get("sd_diff"), decimals=decimals)
        shuf_ci = _fmt_ci(shuf.get("ci_low"), shuf.get("ci_high"), decimals=decimals)
        const_str = _fmt_mean_sd(const.get("mean_diff"), const.get("sd_diff"), decimals=decimals)
        const_ci = _fmt_ci(const.get("ci_low"), const.get("ci_high"), decimals=decimals)

        lines.append(f"{task_label} & {shuf_str} & {shuf_ci} & {const_str} & {const_ci} \\")

    if not has_any:
        lines.append(r"\multicolumn{5}{c}{\textit{No RQ3 timing-intervention records found in master summary JSON.}} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_b200_scaling_table(
    scaling_data: Dict[str, Any],
    venue: str = "neurips",
) -> str:
    """
    Generates Table 3 (RQ4): NVIDIA B200 GPU Throughput, Latency, and Peak Memory Scaling
    across batch sizes B in {1, 16, 256, 4096}.
    """
    rows: List[Dict[str, Any]] = []
    if isinstance(scaling_data, list):
        rows = scaling_data
    elif "records" in scaling_data:
        rows = scaling_data["records"]
    elif "results" in scaling_data:
        rows = scaling_data["results"]

    lines: List[str] = [
        _venue_header_comment(venue),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{RQ4 NVIDIA B200 GPU Inference Throughput, Latency, and Peak Allocated Memory Scaling.} "
        r"Measurements use synchronized CUDA events over 200 timed passes after 30 warmup iterations on resident tensors. "
        r"Throughput is reported in $10^6$ transitions/s ($B \cdot L / t_{\text{batch}}$); Peak Memory denotes PyTorch peak allocated VRAM (MiB). "
        r"Speedup ($\times$) and Memory Saving (\%) are computed relative to each comparator at matched batch size and sequence length.}",
        r"\label{tab:dawin_rq4_b200_scaling}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cc l cccc}",
        r"\toprule",
        r"\textbf{Seq. $L$} & \textbf{Batch $B$} & \textbf{Model Architecture} & \textbf{Latency (ms)} & \textbf{Throughput ($10^6$ trans./s)} & \textbf{Peak VRAM (MiB)} & \textbf{DAWIN Speedup / Mem. Saving} \\",
        r"\midrule",
    ]

    if not rows:
        lines.append(r"\multicolumn{7}{c}{\textit{No B200 profiling records found in scaling JSON.}} \\")
    else:
        # Group by (seq_len, batch_size)
        grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for r in rows:
            key = (int(r.get("seq_len", r.get("L", 32))), int(r.get("batch_size", r.get("B", 256))))
            grouped.setdefault(key, []).append(r)

        sorted_keys = sorted(grouped.keys())
        for g_idx, (seq_l, batch_b) in enumerate(sorted_keys):
            group_rows = grouped[(seq_l, batch_b)]
            # Find DAWIN parallel reference row in this group
            dawin_ref = next(
                (x for x in group_rows if "DAWIN" in str(x.get("model", "")) and "Serial" not in str(x.get("model", ""))),
                None,
            )
            dawin_tp = float(dawin_ref["throughput_m_per_s"]) if dawin_ref and "throughput_m_per_s" in dawin_ref else None
            dawin_mem = float(dawin_ref["peak_mem_mib"]) if dawin_ref and "peak_mem_mib" in dawin_ref else None

            n_grp = len(group_rows)
            for i_r, r in enumerate(group_rows):
                l_col = f"\multirow{{{n_grp}}}{{*}}{{{seq_l:,}}}" if i_r == 0 else ""
                b_col = f"\multirow{{{n_grp}}}{{*}}{{{batch_b:,}}}" if i_r == 0 else ""
                m_raw = str(r.get("model", "unknown"))
                m_label = MODEL_DISPLAY_NAMES.get(m_raw, m_raw.replace("_", r"\_"))

                lat_ms = r.get("latency_ms", r.get("mean_ms"))
                lat_sd = r.get("latency_sd_ms", 0.0)
                tp_m = r.get("throughput_m_per_s")
                mem_mib = r.get("peak_mem_mib")

                is_dawin_par = (dawin_ref is not None and r is dawin_ref)
                lat_str = f"${lat_ms:.3f} \\pm {lat_sd:.3f}$" if lat_ms is not None else r"--"
                tp_str = (
                    f"$\\mathbf{{{tp_m:.3f}}}$" if (is_dawin_par and tp_m is not None)
                    else (f"${tp_m:.3f}$" if tp_m is not None else r"--")
                )
                mem_str = f"${mem_mib:.2f}$" if mem_mib is not None else r"--"

                if is_dawin_par:
                    rel_str = r"\textit{Reference (1.00$\times$ / 0.0\%)}"
                elif dawin_tp is not None and tp_m is not None and tp_m > 0 and dawin_mem is not None and mem_mib is not None and mem_mib > 0:
                    speedup = dawin_tp / float(tp_m)
                    mem_save = 100.0 * (1.0 - dawin_mem / float(mem_mib))
                    rel_str = f"${speedup:.2f}\times$ / ${mem_save:+.1f}\%$"
                else:
                    rel_str = r"--"

                lines.append(f"{l_col} & {b_col} & {m_label} & {lat_str} & {tp_str} & {mem_str} & {rel_str} \\")

            if g_idx < len(sorted_keys) - 1:
                lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DAWIN JSON results to NeurIPS/ICLR/ICML LaTeX tables.")
    parser.add_argument("--results-dir", type=str, default="results_dawin", help="Directory containing JSON summaries.")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--venue", type=str, default="neurips", choices=["neurips", "iclr", "icml"])
    parser.add_argument("--decimals", type=int, default=4, help="Decimal places for accuracy metrics.")
    parser.add_argument("--output", type=str, default="results_dawin/dawin_paper_tables.tex", help="Output .tex path.")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    master_path = res_dir / f"master_benchmark_summary_{args.precision}.json"
    scaling_path = res_dir / f"rq4_b200_scaling_{args.precision}.json"

    master_data: Dict[str, Any] = {}
    if master_path.exists():
        master_data = json.loads(master_path.read_text(encoding="utf-8"))
    else:
        print(f"[Warning] {master_path} not found; generating placeholder Table 1 & Table 2.")

    scaling_data: Dict[str, Any] = {}
    if scaling_path.exists():
        scaling_data = json.loads(scaling_path.read_text(encoding="utf-8"))
    else:
        print(f"[Warning] {scaling_path} not found; generating placeholder Table 3.")

    t1 = generate_main_benchmark_table(master_data, venue=args.venue, decimals=args.decimals)
    t2 = generate_timing_intervention_table(master_data, venue=args.venue, decimals=args.decimals)
    t3 = generate_b200_scaling_table(scaling_data, venue=args.venue)

    full_tex = (
        "% ============================================================================\n"
        "% Auto-generated DAWIN LaTeX Tables (NeurIPS / ICLR / ICML Booktabs Format)\n"
        "% ============================================================================\n\n"
        + t1
        + "\n\n% ----------------------------------------------------------------------------\n\n"
        + t2
        + "\n\n% ----------------------------------------------------------------------------\n\n"
        + t3
        + "\n"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_tex, encoding="utf-8")
    print(f"Successfully exported 3 publication-ready LaTeX table* blocks to: {out_path}")


if __name__ == "__main__":
    main()
