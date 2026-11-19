import os
import json
import shutil
import tempfile
import subprocess
import sys
import torch
import torch.nn as nn

from dawin_core import DAWINLinear
from dawin_models import (
    DAWINMLP,
    DAWINTransformer,
    DAWINGNN,
    MatchedReLUMLP,
    MatchedTimeTransformer,
    MatchedGRUGNN,
    find_matched_width,
)

def test_serial_vs_parallel_scan_equivalence():
    torch.manual_seed(42)
    B, L, D_in, M = 4, 64, 12, 32
    x = torch.randn(B, L, D_in, dtype=torch.float64)
    dt = torch.rand(B, L, dtype=torch.float64) * 2.5
    dt[:, 0] = 0.0
    reset_mask = (torch.rand(B, L) > 0.85).to(torch.float64)

    layer_par = DAWINLinear(D_in, M, mode="quad", use_parallel_scan=True).double()
    layer_ser = DAWINLinear(D_in, M, mode="quad", use_parallel_scan=False).double()
    layer_ser.load_state_dict(layer_par.state_dict())

    out_par = layer_par(x, dt, reset_mask=reset_mask)
    out_ser = layer_ser(x, dt, reset_mask=reset_mask)
    assert torch.allclose(out_par, out_ser, atol=1e-11), f"Max diff: {(out_par - out_ser).abs().max()}"
    print("[PASS] Serial vs. Parallel Associative Scan Equivalence (FP64 max diff < 1e-11)")

def test_bibo_stability_under_extreme_outliers():
    torch.manual_seed(7)
    B, L, D_in, M = 2, 256, 8, 16
    layer = DAWINLinear(D_in, M, mode="quad", gamma_min=1e-2).eval()
    x_extreme = torch.randn(B, L, D_in) * 1e6
    dt = torch.full((B, L), 0.1)
    dt[:, 0] = 0.0

    with torch.no_grad():
        out = layer(x_extreme, dt)
        p, q = out[..., :M], out[..., M:]
        mag = torch.sqrt(p ** 2 + q ** 2)
        dt_min = 0.1
        bound = layer.amplitude / (1.0 - torch.exp(-layer.gamma * dt_min))
        assert torch.isfinite(out).all(), "Non-finite activations under extreme spatial outliers"
        assert (mag <= bound.view(1, 1, M) + 1e-5).all(), "Exceeded Theorem 1 BIBO bound"
    print("[PASS] Theorem 1 BIBO Stability & Outlier Invariance Verified")

def test_backbone_triton_vs_pytorch_parity():
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L, D_in, D_out, M = 2, 48, 8, 4, 16
    x = torch.randn(B, L, D_in, device=device, requires_grad=True)
    dt = torch.rand(B, L, device=device) * 1.5
    dt[:, 0] = 0.0
    dt = dt.detach().requires_grad_(True)

    # 1. DAWINMLP parity
    m_tri = DAWINMLP(D_in, M, D_out, num_layers=2, use_triton=True).to(device)
    m_pt = DAWINMLP(D_in, M, D_out, num_layers=2, use_triton=False).to(device)
    m_pt.load_state_dict(m_tri.state_dict())

    y_tri = m_tri(x, dt)
    y_pt = m_pt(x, dt)
    atol = 1e-4 if device.type == "cuda" else 1e-6
    assert torch.allclose(y_tri, y_pt, atol=atol), "DAWINMLP forward mismatch"

    g_x_tri, g_dt_tri = torch.autograd.grad(y_tri.sum(), (x, dt), retain_graph=True)
    g_x_pt, g_dt_pt = torch.autograd.grad(y_pt.sum(), (x, dt), retain_graph=True)
    assert torch.allclose(g_x_tri, g_x_pt, atol=atol), "DAWINMLP x.grad mismatch"
    assert torch.allclose(g_dt_tri, g_dt_pt, atol=atol), "DAWINMLP dt.grad mismatch"

    y_tri.sum().backward()
    y_pt.sum().backward()
    for (n1, p1), (n2, p2) in zip(m_tri.named_parameters(), m_pt.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=atol), f"DAWINMLP param grad mismatch on {n1}"

    # 2. DAWINTransformer parity
    t_tri = DAWINTransformer(D_in, M, D_out, num_layers=2, num_heads=4, use_triton=True).to(device)
    t_pt = DAWINTransformer(D_in, M, D_out, num_layers=2, num_heads=4, use_triton=False).to(device)
    t_pt.load_state_dict(t_tri.state_dict())

    yt_tri = t_tri(x, dt)
    yt_pt = t_pt(x, dt)
    assert torch.allclose(yt_tri, yt_pt, atol=atol), "DAWINTransformer forward mismatch"
    yt_tri.sum().backward()
    yt_pt.sum().backward()
    for (n1, p1), (n2, p2) in zip(t_tri.named_parameters(), t_pt.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=atol), f"DAWINTransformer param grad mismatch on {n1}"

    # 3. DAWINGNN parity
    N = 5
    xg = torch.randn(B, L, N, D_in, device=device, requires_grad=True)
    dtg = (torch.rand(B, L, N, device=device) * 1.2).requires_grad_(True)
    adj = torch.eye(N, device=device) + 0.25 * torch.ones(N, N, device=device)
    adj = adj / adj.sum(dim=-1, keepdim=True)

    g_tri = DAWINGNN(D_in, M, D_out, num_layers=2, use_triton=True).to(device)
    g_pt = DAWINGNN(D_in, M, D_out, num_layers=2, use_triton=False).to(device)
    g_pt.load_state_dict(g_tri.state_dict())

    yg_tri = g_tri(xg, dtg, adj=adj)
    yg_pt = g_pt(xg, dtg, adj=adj)
    assert torch.allclose(yg_tri, yg_pt, atol=atol), "DAWINGNN forward mismatch"
    yg_tri.sum().backward()
    yg_pt.sum().backward()
    for (n1, p1), (n2, p2) in zip(g_tri.named_parameters(), g_pt.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=atol), f"DAWINGNN param grad mismatch on {n1}"

    print(f"[PASS] Backbone Triton vs. PyTorch Forward & Backward Parity ({device.type})")

def test_parameter_matching_across_backbones():
    D_in, D_out, M = 16, 4, 64
    dawin_mlp = DAWINMLP(D_in, M, D_out, num_layers=2)
    target = sum(p.numel() for p in dawin_mlp.parameters())
    w_mlp, p_mlp = find_matched_width(
        lambda w: MatchedReLUMLP(D_in, w, D_out, num_layers=2), target_params=target
    )
    rel_err = abs(p_mlp - target) / target
    assert rel_err <= 0.02, f"Parameter mismatch {rel_err:.4f} > 0.02"
    print(f"[PASS] Parameter Matching Verified (DAWIN={target}, MatchedMLP={p_mlp}, diff={rel_err*100:.2f}%)")

def test_end_to_end_synthetic_to_latex_and_figures_pipeline():
    tmp_dir = tempfile.mkdtemp(prefix="dawin_e2e_test_")
    try:
        # 1. Run 2-seed synthetic experiment
        cmd_exp = [
            sys.executable,
            "run_dawin_experiments.py",
            "--backbone", "mlp",
            "--task", "damped_oscillator",
            "--comparators", "relu_mlp",
            "--seeds", "101", "102",
            "--epochs", "2",
            "--batch-size", "32",
            "--precision", "fp32",
            "--device", "cpu",
            "--output-dir", tmp_dir,
        ]
        subprocess.run(cmd_exp, check=True)

        summary_path = os.path.join(tmp_dir, "mlp_damped_oscillator_fp32_summary.json")
        assert os.path.exists(summary_path), "Missing task summary JSON"
        with open(summary_path, "r") as f:
            task_summary = json.load(f)

        master_path = os.path.join(tmp_dir, "master_benchmark_summary_fp32.json")
        with open(master_path, "w") as f:
            json.dump({"precision": "fp32", "tasks": [task_summary]}, f, indent=2)

        # 2. Write valid RQ4 B200 scaling fixture
        rq4_fixture = {
            "precision": "fp32",
            "device": "CPU-Test",
            "records": [
                {
                    "seq_len": 32,
                    "batch_size": b,
                    "model": m,
                    "params": 16000,
                    "latency_ms_mean": 1.2 * (b ** 0.4) * mult,
                    "latency_ms_std": 0.05,
                    "throughput_trans_per_sec": (b * 32) / (1.2e-3 * (b ** 0.4) * mult),
                    "peak_memory_mib": 12.0 + 0.05 * b * mult,
                }
                for b in [1, 16, 256, 4096]
                for m, mult in [
                    ("DAWIN_MLP_Parallel", 1.0),
                    ("DAWIN_MLP_Serial", 4.5),
                    ("Time_Transformer", 2.2),
                    ("S5", 1.4),
                ]
            ],
        }
        rq4_path = os.path.join(tmp_dir, "rq4_b200_scaling_fp32.json")
        with open(rq4_path, "w") as f:
            json.dump(rq4_fixture, f, indent=2)

        # 3. Export LaTeX tables
        tex_out = os.path.join(tmp_dir, "dawin_neurips_tables_fp32.tex")
        cmd_tex = [
            sys.executable,
            "export_dawin_latex_tables.py",
            "--results-dir", tmp_dir,
            "--precision", "fp32",
            "--venue", "neurips",
            "--output", tex_out,
        ]
        subprocess.run(cmd_tex, check=True)
        assert os.path.exists(tex_out), "LaTeX .tex output missing"
        with open(tex_out, "r") as f:
            tex_content = f.read()
        for label in ["tab:dawin_main_results", "tab:dawin_rq3_interventions", "tab:dawin_rq4_b200_scaling"]:
            assert label in tex_content, f"Missing LaTeX table {label}"

        # 4. Render PDF and PNG figures
        fig_dir = os.path.join(tmp_dir, "figures")
        cmd_fig = [
            sys.executable,
            "plot_dawin_figures.py",
            "--results-dir", tmp_dir,
            "--precision", "fp32",
            "--seq-len", "32",
            "--formats", "pdf", "png",
            "--output-dir", fig_dir,
        ]
        subprocess.run(cmd_fig, check=True)

        for fig_stem in ["fig_rq1_main_comparison", "fig_rq3_dt_interventions", "fig_rq4_b200_scaling"]:
            for ext in ["pdf", "png"]:
                fpath = os.path.join(fig_dir, f"{fig_stem}.{ext}")
                assert os.path.exists(fpath), f"Missing rendered figure: {fpath}"
                assert os.path.getsize(fpath) > 1024, f"Figure file unexpectedly small: {fpath}"

        print("[PASS] End-to-End Pipeline (Run -> JSON -> LaTeX .tex -> PDF/PNG Figures) Verified")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_serial_vs_parallel_scan_equivalence()
    test_bibo_stability_under_extreme_outliers()
    test_backbone_triton_vs_pytorch_parity()
    test_parameter_matching_across_backbones()
    test_end_to_end_synthetic_to_latex_and_figures_pipeline()
    print("\nAll verify_dawin_harness.py checks passed!")
