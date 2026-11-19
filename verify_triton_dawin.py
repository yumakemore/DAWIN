"""
verify_triton_dawin.py

Numerical verification, mixed-precision (BF16/FP16) autocast stability testing,
and GPU latency/throughput benchmarking for TritonDAWINLinear (both single-block
and multi-block chunked parallel associative scans) against the exact PyTorch
FP32 serial oracle.

Validates:
1. Exact FP32 forward and backward gradient parity across sequence lengths
   L in {32, 2048, 8192, 16384} (single-block & multi-block chunked kernels).
2. Mixed-precision (BF16 & FP16) autocast numerical stability under the NVIDIA B200
   training policy (PyTorch 2.9+/CUDA 13+, BF16 autocast with TF32 disabled on
   matmul and FP32 accumulation inside sensitive recurrence kernels):
   - Zero NaN / Inf values across long horizons (L = 8192), extreme interval
     regimes (burst dt -> 0 and long dropouts dt = 100.0), and spatial outliers
     (||x|| -> 1e5).
   - Bounded quantization error relative to the FP64 exact analytical oracle.
   - Clean gradient flow with finite norms under BF16 autocast and FP16 GradScaler.
3. GPU wall-clock speedup and peak memory profiling up to L = 32768.
"""

from __future__ import annotations

import math
import time
import torch
import torch.nn as nn

from dawin_triton import TritonDAWINLinear, HAS_TRITON


def configure_b200_precision_policy() -> None:
    """
    Mirrors the exact precision policy from the B200 experimental protocol:
    - Matrix-multiplication TF32 disabled
    - cuDNN TF32 enabled
    - Sensitive recurrence state integration retained in FP32
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True


def verify_fp32_parity(device: torch.device) -> None:
    """Verify FP32 forward and backward gradients across single- and multi-block regimes."""
    print(f"[1/3] Running FP32 Forward & Backward Parity Suite on {device}...")
    test_lengths = [32, 2048, 8192, 16384]
    B, D_in, M = 4, 16, 32

    for L in test_lengths:
        torch.manual_seed(42 + L)
        x_base = torch.randn(B, L, D_in, device=device, dtype=torch.float32)
        dt_base = torch.rand(B, L, device=device, dtype=torch.float32) * 0.5 + 0.01
        reset_mask = (torch.rand(B, L, device=device) < 0.05).float()
        reset_mask[:, 0] = 0.0

        serial_layer = TritonDAWINLinear(
            D_in, M, mode="quad", use_triton=False, chunk_size=2048
        ).to(device)
        triton_layer = TritonDAWINLinear(
            D_in, M, mode="quad", use_triton=True, chunk_size=2048
        ).to(device)
        triton_layer.load_state_dict(serial_layer.state_dict())

        x_s = x_base.clone().detach().requires_grad_(True)
        dt_s = dt_base.clone().detach().requires_grad_(True)
        x_t = x_base.clone().detach().requires_grad_(True)
        dt_t = dt_base.clone().detach().requires_grad_(True)

        y_s = serial_layer(x_s, dt_s, reset_mask=reset_mask)
        y_t = triton_layer(x_t, dt_t, reset_mask=reset_mask)

        grad_out = torch.randn_like(y_s)
        y_s.backward(grad_out)
        y_t.backward(grad_out)

        fwd_err = (y_s - y_t).abs().max().item()
        gx_err = (x_s.grad - x_t.grad).abs().max().item()
        gdt_err = (dt_s.grad - dt_t.grad).abs().max().item()

        param_errs = {}
        for (name, p_s), (_, p_t) in zip(
            serial_layer.named_parameters(), triton_layer.named_parameters()
        ):
            param_errs[name] = (p_s.grad - p_t.grad).abs().max().item()

        max_param_err = max(param_errs.values())
        regime = "single-block" if L <= 2048 else f"multi-block ({math.ceil(L / 2048)} chunks)"
        print(
            f"  L={L:5d} [{regime:22s}] | "
            f"fwd={fwd_err:.2e} | gx={gx_err:.2e} | gdt={gdt_err:.2e} | "
            f"max_param={max_param_err:.2e}"
        )

        tol = 5e-4 if L >= 8192 else 2e-4
        assert fwd_err < tol, f"Forward mismatch at L={L}: {fwd_err}"
        assert gx_err < tol, f"Input grad mismatch at L={L}: {gx_err}"
        assert gdt_err < tol, f"dt grad mismatch at L={L}: {gdt_err}"
        assert max_param_err < 5e-3, f"Param grad mismatch at L={L}: {param_errs}"

    print("  -> FP32 parity verified across all sequence lengths.
")


def verify_mixed_precision_autocast(device: torch.device) -> None:
    """
    Validate BF16 and FP16 mixed-precision autocast stability under B200 training conditions:
    1. Long-horizon sequence (L = 8192) across multi-block chunks
    2. Extreme sampling regimes: burst dt = 0.0, long dropout gaps dt = 100.0,
       and extreme spatial outliers (||x|| ~ 1e5)
    3. Comparison against exact FP64 analytical oracle + GradScaler backward check
    """
    print(f"[2/3] Running Mixed-Precision (BF16 / FP16) Autocast Stability Suite on {device}...")
    if device.type != "cuda":
        print("  -> Skipping CUDA autocast suite (CPU device detected).
")
        return

    configure_b200_precision_policy()

    B, L, D_in, M = 4, 8192, 32, 64
    torch.manual_seed(2026)

    # Construct challenging stress-test inputs:
    # - Normal regime + burst zero-intervals + large dropout gaps + spatial outliers
    x_fp64 = torch.randn(B, L, D_in, device=device, dtype=torch.float64)
    x_fp64[:, 100:120, :] *= 1e5  # Extreme spatial outlier burst
    dt_fp64 = torch.rand(B, L, device=device, dtype=torch.float64) * 0.5 + 0.005
    dt_fp64[:, 500:525] = 0.0      # Zero-time boundary burst (dt = 0)
    dt_fp64[:, 1000:1010] = 100.0  # Extreme observation dropout (dt = 100)
    reset_mask = (torch.rand(B, L, device=device) < 0.02).to(torch.float32)
    reset_mask[:, 0] = 0.0

    # Exact FP64 analytical reference
    oracle_fp64 = TritonDAWINLinear(
        D_in, M, mode="quad", use_triton=False, chunk_size=2048
    ).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        y_oracle = oracle_fp64(x_fp64, dt_fp64, reset_mask=reset_mask.to(torch.float64))

    amp_dtypes = [torch.bfloat16]
    if torch.cuda.is_available():
        amp_dtypes.append(torch.float16)

    for amp_dtype in amp_dtypes:
        dtype_name = "BF16" if amp_dtype == torch.bfloat16 else "FP16"
        model = TritonDAWINLinear(
            D_in, M, mode="quad", use_triton=True, chunk_size=2048
        ).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            model.load_state_dict({k: v.float() for k, v in oracle_fp64.state_dict().items()})

        x_in = x_fp64.to(dtype=amp_dtype).detach().requires_grad_(True)
        dt_in = dt_fp64.to(dtype=amp_dtype).detach().requires_grad_(True)

        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            y_amp = model(x_in, dt_in, reset_mask=reset_mask)
            loss = (y_amp.float() ** 2).mean()

        assert torch.isfinite(y_amp).all(), f"{dtype_name} forward produced NaN or Inf!"
        assert torch.isfinite(loss), f"{dtype_name} loss produced NaN or Inf!"

        # Verify BIBO boundedness even under ||x|| ~ 1e5 and dt = 0 bursts
        max_state_mag = y_amp.float().abs().max().item()
        max_amp_param = model.amplitude.max().item()
        assert max_state_mag < L * max_amp_param + 1e-2, (
            f"{dtype_name} violated BIBO bound: {max_state_mag}"
        )

        scaler.scale(loss).backward()
        scaler.unscale_(torch.optim.AdamW(model.parameters(), lr=1e-3))

        assert x_in.grad is not None and torch.isfinite(x_in.grad).all(), (
            f"{dtype_name} x.grad contains NaN/Inf!"
        )
        assert dt_in.grad is not None and torch.isfinite(dt_in.grad).all(), (
            f"{dtype_name} dt.grad contains NaN/Inf!"
        )

        grad_norms = {}
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (
                f"{dtype_name} parameter '{name}' grad contains NaN/Inf!"
            )
            grad_norms[name] = p.grad.norm().item()

        # Measure relative error against FP64 oracle on non-outlier normal regime
        normal_slice = slice(2000, 4000)
        rel_err = (
            (y_amp[:, normal_slice].float() - y_oracle[:, normal_slice].float()).norm()
            / (y_oracle[:, normal_slice].float().norm() + 1e-8)
        ).item()

        print(
            f"  Autocast {dtype_name:4s} (L={L}, outliers+dt=0+dt=100) | "
            f"finite=PASS | max|z|={max_state_mag:.3f} | "
            f"rel_err_vs_FP64={rel_err:.3e} | "
            f"grad_norm(k)={grad_norms['k']:.3e}, grad_norm(gamma)={grad_norms['raw_gamma']:.3e}"
        )

        # Because input phase phi = k @ x wraps modulo 2*pi when ||x|| ~ 1e5 in 16-bit,
        # normal-regime accuracy stays tightly bounded while outlier bursts stay finite.
        tol_rel = 2.5e-2 if amp_dtype == torch.bfloat16 else 1.0e-2
        assert rel_err < tol_rel, (
            f"{dtype_name} relative error vs FP64 exceeded tolerance: {rel_err} >= {tol_rel}"
        )

    print("  -> Mixed-precision (BF16/FP16) autocast stability verified.
")


def benchmark_triton_vs_serial(device: torch.device) -> None:
    """Benchmark forward+backward latency and peak GPU memory under BF16 autocast."""
    print(f"[3/3] Benchmarking Fused Triton Scan vs. Serial PyTorch on {device} (BF16 Policy)...")
    if device.type != "cuda":
        print("  -> Skipping GPU throughput benchmark (CPU device detected).
")
        return

    configure_b200_precision_policy()
    B, D_in, M = 32, 128, 128
    lengths = [512, 4096, 16384, 32768]

    for L in lengths:
        x = torch.randn(B, L, D_in, device=device, dtype=torch.bfloat16, requires_grad=True)
        dt = torch.rand(B, L, device=device, dtype=torch.bfloat16) * 0.2 + 0.01

        triton_mod = TritonDAWINLinear(
            D_in, M, mode="quad", use_triton=True, chunk_size=2048
        ).to(device)

        # Warmup
        for _ in range(5):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = triton_mod(x, dt)
                loss = out.sum()
            loss.backward()
            triton_mod.zero_grad(set_to_none=True)
            x.grad = None

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = triton_mod(x, dt)
                loss = out.sum()
            loss.backward()
            triton_mod.zero_grad(set_to_none=True)
            x.grad = None
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) * 1000.0 / iters
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        transitions_per_sec = (B * L) / (triton_ms / 1000.0) / 1e6

        if L <= 4096:
            serial_mod = TritonDAWINLinear(
                D_in, M, mode="quad", use_triton=False, chunk_size=2048
            ).to(device)
            for _ in range(3):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_s = serial_mod(x, dt)
                    loss_s = out_s.sum()
                loss_s.backward()
                serial_mod.zero_grad(set_to_none=True)
                x.grad = None
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(5):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_s = serial_mod(x, dt)
                    loss_s = out_s.sum()
                loss_s.backward()
                serial_mod.zero_grad(set_to_none=True)
                x.grad = None
            torch.cuda.synchronize()
            serial_ms = (time.perf_counter() - t1) * 1000.0 / 5
            speedup_str = f"serial={serial_ms:7.2f} ms | speedup={serial_ms / triton_ms:5.1f}x"
        else:
            speedup_str = "serial=skipped (L>4096)"

        print(
            f"  B={B:2d}, L={L:5d}, M={M:3d} | "
            f"Triton fwd+bwd={triton_ms:6.2f} ms ({transitions_per_sec:6.2f}M trans/s) | "
            f"peak={peak_mib:6.1f} MiB | {speedup_str}"
        )


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DAWIN Triton & Mixed-Precision Verification (HAS_TRITON={HAS_TRITON}) ===")
    verify_fp32_parity(dev)
    verify_mixed_precision_autocast(dev)
    benchmark_triton_vs_serial(dev)
    print("All verification and stability checks passed successfully.")
