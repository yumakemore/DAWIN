# Damped Atomic Wave Integration Neurons (DAWIN)

[![License: MIT](https://shields.io)](LICENSE)
[![Python 3.10+](https://shields.io)](https://python.org)
[![PyTorch 2.4+](https://shields.io)](https://pytorch.org)
[![Triton 3.0+](https://shields.io)](https://triton-lang.org)
[![Open In Colab](https://google.com)](https://google.com)

Official PyTorch and Triton reference implementation for **"Damped Atomic Wave Integration Neurons (DAWIN): A Universal Primitive for Continuous-Time Spatiotemporal Learning"** by Byunggu Yu and Junwhan Kim (University of the District of Columbia).

DAWIN serves as a drop-in, architecture-agnostic replacement for the conventional perceptron \(y = \sigma(\mathbf{w}^\top \mathbf{x} + b)\). For each unit m, DAWIN combines spatial wave phase and continuous-time Zero-Order Hold (ZOH) interval scaling into an indivisible (*atomic*) complex wave packet \(u_{i,m}\) and integrates it into a neuron-local state \(\psi_{i,m} \in \mathbb{C}\) via:

\[u_{i,m} = A_m \alpha_m(\Delta t_i) \exp\big(i(\mathbf{k}_m^\top \mathbf{x}_i - \theta_m)\big), \qquad \psi_{i,m} = \exp\big(-(\gamma_m + i\omega_m)\Delta t_i\big)\psi_{i-1,m} + u_{i,m}\]

where \(\alpha_m(\Delta t_i) = 1 - e^{-\gamma_m \Delta t_i}\) for \(\Delta t_i > 0\) and \(\alpha_m(0) = 1\).

---

## Drop-In Usage Example

`DAWINLinear` (and its fused CUDA kernel `TritonDAWINLinear`) replaces `nn.Linear + Activation` anywhere in a model while propagating continuous-time complex quadrature states across irregular intervals \(\Delta t_i\):

```python
import torch
from dawin_core import DAWINLinear

# Batch size B=8, sequence length L=128, input dim d_in=16, hidden units M=64
x = torch.randn(8, 128, 16, device="cuda")
dt = torch.rand(8, 128, device="cuda") * 0.5  # irregular elapsed intervals

# Hidden layer emits paired real/imaginary quadratures [B, L, 2*M]
hidden_layer = DAWINLinear(in_features=16, out_features=64, mode="quad").cuda()
# Readout layer emits real quadrature [B, L, M]
out_layer = DAWINLinear(in_features=128, out_features=4, mode="real").cuda()

h = hidden_layer(x, dt)                        # [8, 128, 128]
y = out_layer(h, dt)                           # [8, 128, 4]
reg_loss = hidden_layer.laplacian_penalty()    # closed-form spatiotemporal curvature penalty
```

---

## Repository Structure

### Core Model, Kernel & Dataset Modules
* `dawin_core.py`: Drop-in `DAWINLinear` primitive with paired-real quadrature propagation, parallel associative prefix scan (`torch._higher_order_ops.associative_scan`), exact serial fallback, component ablation toggles (`none`, `undamped`, `memoryless`, `cosine_only`, `decoupled`), and closed-form spatiotemporal Laplacian curvature regularization (`laplacian_penalty()`, `compute_dawin_laplacian_loss()`).
* `dawin_triton.py`: Multi-block chunked Triton parallel associative scan kernel and custom autograd bridge supporting arbitrarily long sequences ($L > 4096$) in $O(\log L)$ parallel depth without materializing intermediate complex $[B, L, M]$ tensors in HBM.
* `dawin_models.py`: Drop-in continuous-time backbones (`DAWINMLP`, `DAWINTransformer`, `DAWINGNN`) with automatic CUDA Triton dispatch (`use_triton=True`), alongside parameter-matched comparators (`MatchedReLUMLP`, `MatchedTimeTransformer`, `MatchedNeuralCDE`, `MatchedS5`, `MatchedMamba`, `MatchedGRUGNN`) and `find_matched_width` ($\le 2\%$ parameter parity).
* `dawin_datasets.py`: Standardized data loaders, training-only normalization pipelines, and task metrics for MuJoCo irregular dynamics (`HalfCheetah-v5`, `Hopper-v5`, `Reacher-v5`, `Walker2d-v5`), PhysioNet Challenge 2012 (mask-aware mortality, mask-aware imputation, and dense WSP displacement), and ETH/UCY + JAAD trajectories.
* `dawin_diagnostics.py`: RQ3 $\Delta t$-intervention diagnostics (true, batch-shuffled $\Delta t_{\text{perm}}$, and constant-mean $\Delta t_{\text{const}}$), RQ4 GPU throughput/memory profiling, and 20,000-resample paired bootstrap / Holm-Bonferroni significance tests.

### Experiment Runners, Exporters & Plotting Scripts
* `run_dawin_experiments.py`: Executes 10-seed (`101`–`110`) parameter-matched sweeps for a single task with configurable `--precision` (`fp32`, `bf16`, `fp16`) and writes `.json`, `.csv`, and `.md` summaries.
* `run_all_dawin_benchmarks.py`: Orchestrates sequential 10-seed sweeps across all 11 benchmark tasks, runs the `--profile-b200` RQ4 GPU scaling suite, and automatically compiles LaTeX tables and publication figures.
* `export_dawin_latex_tables.py`: Compiles `master_benchmark_summary_<precision>.json` and `rq4_b200_scaling_<precision>.json` into venue-agnostic (`generic`, outputting `dawin_tables_<precision>.tex`) or venue-specific (`neurips`, `iclr`, `icml`) LaTeX `booktabs` tables.
* `plot_dawin_figures.py`: Renders publication-ready vector (`.pdf`) and raster (`.png`) figures for RQ1/RQ2 accuracy comparisons, RQ3 causal timing interventions, and RQ4 NVIDIA B200 GPU scaling curves.

### Verification Suites & Interactive Notebook
* `verify_dawin_harness.py`: Validates serial-vs-scan equivalence, BIBO stability under spatial outliers ($\|\mathbf{x}\| \sim 10^6$), parameter parity, Triton-vs-PyTorch backbone gradient parity, and the end-to-end synthetic-to-LaTeX-and-figures pipeline.
* `verify_extended_comparators.py`: Validates forward shapes, parameter parity ($\le 2\%$), and gradient flow across all parameter-matched comparators.
* `verify_dawin_datasets.py`: Validates tensor shapes, non-negative $\Delta t$ invariants, and evaluation metrics across all 11 benchmark configurations.
* `verify_triton_dawin.py`: Verifies numerical parity of the multi-block Triton kernel under `FP32`, `BF16`, and `FP16` autocast and benchmarks GPU latency.
* [`DAWIN_10seed.ipynb`](DAWIN_10seed.ipynb): Self-contained Jupyter/Colab notebook for interactive GPU execution and result inspection.

---

## Installation & Runtime Policy

```bash
git clone https://github.com
cd DAWIN
pip install -r requirements.txt
```

Following our NVIDIA B200 experimental protocol, mixed-precision runs default to `--precision bf16` with matrix-multiplication TF32 disabled (`torch.backends.cuda.matmul.allow_tf32 = False`), cuDNN TF32 enabled (`torch.backends.cudnn.allow_tf32 = True`), and FP32 accumulation retained inside the associative scan recurrence.

---

## Quickstart & Verification

Run all four verification suites prior to launching cluster experiments:

```bash
python verify_dawin_harness.py
python verify_extended_comparators.py
python verify_dawin_datasets.py
python verify_triton_dawin.py
```

---

## Interactive Execution (Google Colab)

For interactive, cell-by-cell execution on cloud GPUs (`B200`, `B100`, `G4`, `A100`, `L4`, or `T4`), open [`DAWIN_10seed.ipynb`](DAWIN_10seed.ipynb) directly in Google Colab via the badge at the top of this page.

---

## Reproducing Benchmarks, Tables & Figures

All benchmark, profiling, and export commands save outputs to `./results_dawin` by default (override with `--results-dir <PATH>`).

### 1. Full 11-Task 10-Seed Benchmark Sweep
```bash
python run_all_dawin_benchmarks.py --precision fp32 --laplacian-weight 1e-4 --device cuda:0
```

### 2. Standalone RQ4 GPU Throughput & Memory Profiling (BF16)
```bash
python run_all_dawin_benchmarks.py --profile-only --precision bf16 --device cuda:0
```

### 3. Single-Task 10-Seed Sweep & Manual Table/Figure Export
```bash
python run_all_dawin_benchmarks.py --tasks HalfCheetah-v5 --precision fp32 --laplacian-weight 1e-4 --force
python export_dawin_latex_tables.py --results-dir results_dawin --precision fp32 --venue generic
python plot_dawin_figures.py --results-dir results_dawin --precision fp32 --formats pdf png
```

### Generated Output Artifacts
Running the benchmark and export scripts populates `./results_dawin` with:
* `master_benchmark_summary_<precision>.{json,csv,md}`: Aggregated 10-seed Mean ± SD, paired differences, 95% bootstrap CIs, Holm-Bonferroni p-values, and RQ3 $\Delta t$-intervention results.
* `rq4_b200_scaling_<precision>.{json,csv,md}`: GPU latency (`ms`), transition throughput (`transitions/s`), and peak VRAM (`MiB`) across batch sizes `1, 16, 256, 4096`.
* `dawin_tables_<precision>.tex`: Ready-to-include LaTeX `booktabs` tables (`\begin{table*}`).
* `figures/fig_rq*.{pdf,png}`: Publication-ready accuracy comparison, timing-intervention, and scaling figures.

---

## Citation

If you find DAWIN useful in your research, please cite:

```bibtex
@article{yu2026dawin,
  title   = {Damped Atomic Wave Integration Neurons (DAWIN): A Universal Primitive for Continuous-Time Spatiotemporal Learning},
  author  = {Yu, Byunggu and Kim, Junwhan},
  year    = {2026},
  journal = {arXiv preprint}
}
```

## License

This project is released under the [MIT License](LICENSE).
