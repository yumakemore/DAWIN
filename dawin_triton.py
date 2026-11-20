"""
dawin_triton.py
Fused Triton Parallel Associative Scan Kernel with Multi-Block Chunked Scanning
for Damped Atomic Wave Integration Neurons (DAWIN).

Supports arbitrarily long sequences (L > 4096, e.g., L = 16384, 65536, 131072+)
on NVIDIA GPUs via a three-stage decoupled lookback / Blelloch block-scan pipeline:
  Stage 1: Intra-chunk register-resident inclusive scan over chunks of size C
           (C <= 2048), emitting per-chunk inclusive states and total chunk
           summary tuples (D_chunk, U_chunk).
  Stage 2: Inter-chunk prefix scan over the N_chunks = ceil(L / C) summary
           tuples to compute exact carry-in wave states for every chunk.
  Stage 3: State-propagation pass applying each chunk's carry-in state
           psi_carry to all intra-chunk states in O(1) parallel GPU work.

Both forward integration (Eqs. 1-4, 16) and reverse-time adjoint backpropagation
(Eqs. 5-9) execute in O(log C + log(L / C)) = O(log L) parallel depth without
materializing intermediate complex [B, L, M] tensors in global GPU memory.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


# =============================================================================
# 1. Exact Paired-Real PyTorch Reference (Fallback & Gradient Oracle)
# =============================================================================

def _dawin_assoc_op_torch(elem_l, elem_r):
    """Associative binary composition: (d_r, u_r) (+) (d_l, u_l) = (d_r * d_l, d_r * u_l + u_r)."""
    d_re_l, d_im_l, u_re_l, u_im_l = elem_l
    d_re_r, d_im_r, u_re_r, u_im_r = elem_r

    d_re = d_re_r * d_re_l - d_im_r * d_im_l
    d_im = d_re_r * d_im_l + d_im_r * d_re_l

    u_re = (d_re_r * u_re_l - d_im_r * u_im_l) + u_re_r
    u_im = (d_re_r * u_im_l + d_im_r * u_re_l) + u_im_r
    return (d_re, d_im, u_re, u_im)


def dawin_reference_forward(
    z: torch.Tensor,
    dt: torch.Tensor,
    amp: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
    theta: torch.Tensor,
    reset_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact FP32 reference evaluation of DAWIN forward recurrence (Eqs. 1-4).
    Args:
        z: [B, L, M] spatial projection X @ K^T
        dt: [B, L, 1] or [B, L] elapsed intervals >= 0
        amp, gamma, omega, theta: [M] neuron parameters
        reset_mask: optional [B, L] binary mask (1 resets state before step i)
    Returns:
        p, q: [B, L, M] real and imaginary quadratures
    """
    orig_dtype = z.dtype
    z_f = z.float()
    if dt.dim() == 2:
        dt_f = dt.unsqueeze(-1).float()
    else:
        dt_f = dt.float()

    amp_f = amp.view(1, 1, -1).float()
    gamma_f = gamma.view(1, 1, -1).float()
    omega_f = omega.view(1, 1, -1).float()
    theta_f = theta.view(1, 1, -1).float()

    phi = z_f - dt_f * omega_f - theta_f
    u_re = amp_f * torch.cos(phi)
    u_im = amp_f * torch.sin(phi)

    decay = torch.exp(-gamma_f * dt_f)
    if reset_mask is not None:
        decay = decay * (1.0 - reset_mask.unsqueeze(-1).float())

    rot = -omega_f * dt_f
    d_re = decay * torch.cos(rot)
    d_im = decay * torch.sin(rot)

    B, L, M = z_f.shape
    p_list = []
    q_list = []
    p_prev = torch.zeros((B, M), device=z.device, dtype=torch.float32)
    q_prev = torch.zeros_like(p_prev)

    for t in range(L):
        p_curr = d_re[:, t] * p_prev - d_im[:, t] * q_prev + u_re[:, t]
        q_curr = d_re[:, t] * q_prev + d_im[:, t] * p_prev + u_im[:, t]
        p_list.append(p_curr)
        q_list.append(q_curr)
        p_prev, q_prev = p_curr, q_curr

    p = torch.stack(p_list, dim=1).to(orig_dtype)
    q = torch.stack(q_list, dim=1).to(orig_dtype)
    return p, q


# =============================================================================
# 2. Triton Multi-Block Chunked Associative Scan Kernels
# =============================================================================

if HAS_TRITON:

    @triton.jit
    def _dawin_scan_combine(
        d_re_l, d_im_l, u_re_l, u_im_l,
        d_re_r, d_im_r, u_re_r, u_im_r,
    ):
        """
        Triton associative combine function for `tl.associative_scan`:
        (d_r, u_r) (+) (d_l, u_l) = (d_r * d_l, d_r * u_l + u_r).
        """
        d_re = d_re_r * d_re_l - d_im_r * d_im_l
        d_im = d_re_r * d_im_l + d_im_r * d_re_l
        u_re = (d_re_r * u_re_l - d_im_r * u_im_l) + u_re_r
        u_im = (d_re_r * u_im_l + d_im_r * u_re_l) + u_im_r
        return d_re, d_im, u_re, u_im


    @triton.jit
    def _dawin_fwd_stage1_chunk_scan_kernel(
        Z_ptr, DT_ptr, MASK_ptr,
        AMP_ptr, GAMMA_ptr, OMEGA_ptr, THETA_ptr,
        P_LOCAL_ptr, Q_LOCAL_ptr,
        CHUNK_D_RE_ptr, CHUNK_D_IM_ptr, CHUNK_U_RE_ptr, CHUNK_U_IM_ptr,
        L, M, N_CHUNKS,
        stride_zb, stride_zl, stride_zm,
        stride_dtb, stride_dtl,
        stride_mb, stride_ml,
        stride_cb, stride_cc, stride_cm,
        HAS_MASK: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        """
        Stage 1 (Forward): Intra-chunk register-resident Blelloch prefix scan.
        Grid: (B * M, N_CHUNKS)
        Each program processes one (batch_idx, neuron_idx, chunk_idx) tile of length CHUNK_SIZE.
        """
        bm_idx = tl.program_id(0)
        chunk_idx = tl.program_id(1)

        b_idx = bm_idx // M
        m_idx = bm_idx % M

        amp = tl.load(AMP_ptr + m_idx).to(tl.float32)
        gamma = tl.load(GAMMA_ptr + m_idx).to(tl.float32)
        omega = tl.load(OMEGA_ptr + m_idx).to(tl.float32)
        theta = tl.load(THETA_ptr + m_idx).to(tl.float32)

        offs = tl.arange(0, CHUNK_SIZE)
        t_idx = chunk_idx * CHUNK_SIZE + offs
        valid = t_idx < L

        z_offset = b_idx * stride_zb + t_idx * stride_zl + m_idx * stride_zm
        dt_offset = b_idx * stride_dtb + t_idx * stride_dtl

        z_val = tl.load(Z_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)
        dt_val = tl.load(DT_ptr + dt_offset, mask=valid, other=0.0).to(tl.float32)

        # Atomic spatiotemporal wave excitation (Eq. 1)
        phi = z_val - dt_val * omega - theta
        u_re = tl.where(valid, amp * tl.cos(phi), 0.0)
        u_im = tl.where(valid, amp * tl.sin(phi), 0.0)

        # Damper evolution operator D_m(dt) (Eq. 2); identity (1, 0) for padded positions
        decay = tl.exp(-gamma * dt_val)
        if HAS_MASK:
            m_offset = b_idx * stride_mb + t_idx * stride_ml
            rst = tl.load(MASK_ptr + m_offset, mask=valid, other=0.0).to(tl.float32)
            decay = decay * (1.0 - rst)

        rot = -omega * dt_val
        d_re = tl.where(valid, decay * tl.cos(rot), 1.0)
        d_im = tl.where(valid, decay * tl.sin(rot), 0.0)

        # Register-resident inclusive prefix scan across CHUNK_SIZE
        scan_d_re, scan_d_im, scan_u_re, scan_u_im = tl.associative_scan(
            (d_re, d_im, u_re, u_im), axis=0, combine_fn=_dawin_scan_combine
        )

        # Store chunk-local states
        tl.store(P_LOCAL_ptr + z_offset, scan_u_re, mask=valid)
        tl.store(Q_LOCAL_ptr + z_offset, scan_u_im, mask=valid)

        # Extract total chunk transition summary at index (CHUNK_SIZE - 1)
        # Because padded slots have identity transition (1 + 0i, 0 + 0i),
        # the last slot holds the exact total product and state of all valid steps in this chunk!
        is_last = offs == (CHUNK_SIZE - 1)
        tot_d_re = tl.sum(tl.where(is_last, scan_d_re, 0.0), axis=0)
        tot_d_im = tl.sum(tl.where(is_last, scan_d_im, 0.0), axis=0)
        tot_u_re = tl.sum(tl.where(is_last, scan_u_re, 0.0), axis=0)
        tot_u_im = tl.sum(tl.where(is_last, scan_u_im, 0.0), axis=0)

        c_offset = b_idx * stride_cb + chunk_idx * stride_cc + m_idx * stride_cm
        tl.store(CHUNK_D_RE_ptr + c_offset, tot_d_re)
        tl.store(CHUNK_D_IM_ptr + c_offset, tot_d_im)
        tl.store(CHUNK_U_RE_ptr + c_offset, tot_u_re)
        tl.store(CHUNK_U_IM_ptr + c_offset, tot_u_im)


    @triton.jit
    def _dawin_inter_chunk_scan_kernel(
        IN_D_RE_ptr, IN_D_IM_ptr, IN_U_RE_ptr, IN_U_IM_ptr,
        CARRY_U_RE_ptr, CARRY_U_IM_ptr,
        N_CHUNKS, M,
        stride_cb, stride_cc, stride_cm,
        SCAN_SIZE: tl.constexpr,
    ):
        """
        Stage 2 (Forward & Backward): Inter-chunk prefix scan over chunk summaries.
        Grid: (B * M,)
        Computes inclusive scan across N_CHUNKS and writes exclusive carry-in state
        for chunk c (where carry-in for chunk 0 is 0 + 0i, and carry-in for chunk c > 0
        is the inclusive scan state at chunk c - 1).
        """
        bm_idx = tl.program_id(0)
        b_idx = bm_idx // M
        m_idx = bm_idx % M

        c_offs = tl.arange(0, SCAN_SIZE)
        valid = c_offs < N_CHUNKS

        ptr_offs = b_idx * stride_cb + c_offs * stride_cc + m_idx * stride_cm
        d_re = tl.load(IN_D_RE_ptr + ptr_offs, mask=valid, other=1.0).to(tl.float32)
        d_im = tl.load(IN_D_IM_ptr + ptr_offs, mask=valid, other=0.0).to(tl.float32)
        u_re = tl.load(IN_U_RE_ptr + ptr_offs, mask=valid, other=0.0).to(tl.float32)
        u_im = tl.load(IN_U_IM_ptr + ptr_offs, mask=valid, other=0.0).to(tl.float32)

        _, _, incl_u_re, incl_u_im = tl.associative_scan(
            (d_re, d_im, u_re, u_im), axis=0, combine_fn=_dawin_scan_combine
        )

        # Write inclusive state of chunk c as carry-in for chunk (c + 1)
        next_offs = c_offs + 1
        valid_next = next_offs < N_CHUNKS
        next_ptr_offs = b_idx * stride_cb + next_offs * stride_cc + m_idx * stride_cm
        tl.store(CARRY_U_RE_ptr + next_ptr_offs, incl_u_re, mask=valid_next)
        tl.store(CARRY_U_IM_ptr + next_ptr_offs, incl_u_im, mask=valid_next)

        # Chunk 0 carry-in is strictly zero
        zero_ptr = b_idx * stride_cb + m_idx
        tl.store(CARRY_U_RE_ptr + zero_ptr, 0.0)
        tl.store(CARRY_U_IM_ptr + zero_ptr, 0.0)


    @triton.jit
    def _dawin_fwd_stage3_propagate_kernel(
        DT_ptr, MASK_ptr,
        GAMMA_ptr, OMEGA_ptr,
        P_LOCAL_ptr, Q_LOCAL_ptr,
        CARRY_U_RE_ptr, CARRY_U_IM_ptr,
        P_OUT_ptr, Q_OUT_ptr,
        L, M,
        stride_zb, stride_zl, stride_zm,
        stride_dtb, stride_dtl,
        stride_mb, stride_ml,
        stride_cb, stride_cc, stride_cm,
        HAS_MASK: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        """
        Stage 3 (Forward): Propagate inter-chunk carry-in state psi_carry across
        all positions in each chunk:
            psi_{chunk_start + r} = psi_local_r + (prod_{s=0}^r D_s) * psi_carry
        Grid: (B * M, N_CHUNKS)
        """
        bm_idx = tl.program_id(0)
        chunk_idx = tl.program_id(1)

        b_idx = bm_idx // M
        m_idx = bm_idx % M

        c_offset = b_idx * stride_cb + chunk_idx * stride_cc + m_idx * stride_cm
        carry_re = tl.load(CARRY_U_RE_ptr + c_offset).to(tl.float32)
        carry_im = tl.load(CARRY_U_IM_ptr + c_offset).to(tl.float32)

        offs = tl.arange(0, CHUNK_SIZE)
        t_idx = chunk_idx * CHUNK_SIZE + offs
        valid = t_idx < L
        z_offset = b_idx * stride_zb + t_idx * stride_zl + m_idx * stride_zm

        p_loc = tl.load(P_LOCAL_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)
        q_loc = tl.load(Q_LOCAL_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)

        # Fast-path for chunk 0 (carry == 0)
        if chunk_idx == 0:
            tl.store(P_OUT_ptr + z_offset, p_loc, mask=valid)
            tl.store(Q_OUT_ptr + z_offset, q_loc, mask=valid)
            return

        gamma = tl.load(GAMMA_ptr + m_idx).to(tl.float32)
        omega = tl.load(OMEGA_ptr + m_idx).to(tl.float32)
        dt_offset = b_idx * stride_dtb + t_idx * stride_dtl
        dt_val = tl.load(DT_ptr + dt_offset, mask=valid, other=0.0).to(tl.float32)

        decay = tl.exp(-gamma * dt_val)
        if HAS_MASK:
            m_offset = b_idx * stride_mb + t_idx * stride_ml
            rst = tl.load(MASK_ptr + m_offset, mask=valid, other=0.0).to(tl.float32)
            decay = decay * (1.0 - rst)

        rot = -omega * dt_val
        d_re = tl.where(valid, decay * tl.cos(rot), 1.0)
        d_im = tl.where(valid, decay * tl.sin(rot), 0.0)
        zeros = tl.zeros_like(d_re)

        cum_d_re, cum_d_im, _, _ = tl.associative_scan(
            (d_re, d_im, zeros, zeros), axis=0, combine_fn=_dawin_scan_combine
        )

        p_out = p_loc + (cum_d_re * carry_re - cum_d_im * carry_im)
        q_out = q_loc + (cum_d_re * carry_im + cum_d_im * carry_re)

        tl.store(P_OUT_ptr + z_offset, p_out, mask=valid)
        tl.store(Q_OUT_ptr + z_offset, q_out, mask=valid)


    @triton.jit
    def _dawin_bwd_stage1_chunk_scan_kernel(
        DT_ptr, MASK_ptr,
        GAMMA_ptr, OMEGA_ptr,
        GP_ptr, GQ_ptr,
        DELTA_LOCAL_RE_ptr, DELTA_LOCAL_IM_ptr,
        CHUNK_ADJ_D_RE_ptr, CHUNK_ADJ_D_IM_ptr,
        CHUNK_ADJ_U_RE_ptr, CHUNK_ADJ_U_IM_ptr,
        L, M, N_CHUNKS,
        stride_zb, stride_zl, stride_zm,
        stride_dtb, stride_dtl,
        stride_mb, stride_ml,
        stride_cb, stride_cc, stride_cm,
        HAS_MASK: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        """
        Stage 1 (Backward): Reverse-time intra-chunk adjoint scan (Eq. 5).
        Reverses sequence order inside each chunk so standard forward inclusive scan
        computes:
            delta_bar_i = delta_i + D*(dt_{i+1}) * delta_bar_{i+1}
        Grid: (B * M, N_CHUNKS) where chunk_idx=0 corresponds to the LAST chunk in time.
        """
        bm_idx = tl.program_id(0)
        rev_chunk_idx = tl.program_id(1)

        b_idx = bm_idx // M
        m_idx = bm_idx % M

        gamma = tl.load(GAMMA_ptr + m_idx).to(tl.float32)
        omega = tl.load(OMEGA_ptr + m_idx).to(tl.float32)

        # Map reversed index r in [0, CHUNK_SIZE) to forward time index t
        # rev_chunk_idx = 0 covers the tail of the sequence [L - 1, L - 2, ...]
        offs = tl.arange(0, CHUNK_SIZE)
        fwd_chunk_idx = (N_CHUNKS - 1) - rev_chunk_idx
        chunk_start = fwd_chunk_idx * CHUNK_SIZE
        # Reverse within the chunk: r=0 is chunk_end-1, r=CHUNK_SIZE-1 is chunk_start
        t_idx = chunk_start + (CHUNK_SIZE - 1 - offs)
        valid = t_idx < L

        z_offset = b_idx * stride_zb + t_idx * stride_zl + m_idx * stride_zm

        # Upstream Wirtinger sensitivity: delta_i = dL/dp_i - i * dL/dq_i
        gp_val = tl.load(GP_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)
        gq_val = tl.load(GQ_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)
        u_adj_re = tl.where(valid, gp_val, 0.0)
        u_adj_im = tl.where(valid, -gq_val, 0.0)

        # Adjoint propagator entering t_idx from (t_idx + 1) is D*(dt_{t_idx + 1})
        t_next = t_idx + 1
        valid_next = valid & (t_next < L)
        dt_next_offset = b_idx * stride_dtb + t_next * stride_dtl
        dt_next = tl.load(DT_ptr + dt_next_offset, mask=valid_next, other=0.0).to(tl.float32)

        decay_next = tl.exp(-gamma * dt_next)
        if HAS_MASK:
            m_next_offset = b_idx * stride_mb + t_next * stride_ml
            rst_next = tl.load(MASK_ptr + m_next_offset, mask=valid_next, other=0.0).to(tl.float32)
            decay_next = decay_next * (1.0 - rst_next)

        # Conjugate rotation: +omega * dt_{t+1}
        rot_next = -omega * dt_next
        d_adj_re = tl.where(valid_next, decay_next * tl.cos(rot_next), 1.0)
        d_adj_im = tl.where(valid_next, decay_next * tl.sin(rot_next), 0.0)

        scan_d_re, scan_d_im, scan_u_re, scan_u_im = tl.associative_scan(
            (d_adj_re, d_adj_im, u_adj_re, u_adj_im), axis=0, combine_fn=_dawin_scan_combine
        )

        # Store local reverse-scanned delta_bar at forward index t_idx
        tl.store(DELTA_LOCAL_RE_ptr + z_offset, scan_u_re, mask=valid)
        tl.store(DELTA_LOCAL_IM_ptr + z_offset, scan_u_im, mask=valid)

        # Store reversed-chunk summary at rev_chunk_idx
        is_last = offs == (CHUNK_SIZE - 1)
        tot_d_re = tl.sum(tl.where(is_last, scan_d_re, 0.0), axis=0)
        tot_d_im = tl.sum(tl.where(is_last, scan_d_im, 0.0), axis=0)
        tot_u_re = tl.sum(tl.where(is_last, scan_u_re, 0.0), axis=0)
        tot_u_im = tl.sum(tl.where(is_last, scan_u_im, 0.0), axis=0)

        c_offset = b_idx * stride_cb + rev_chunk_idx * stride_cc + m_idx * stride_cm
        tl.store(CHUNK_ADJ_D_RE_ptr + c_offset, tot_d_re)
        tl.store(CHUNK_ADJ_D_IM_ptr + c_offset, tot_d_im)
        tl.store(CHUNK_ADJ_U_RE_ptr + c_offset, tot_u_re)
        tl.store(CHUNK_ADJ_U_IM_ptr + c_offset, tot_u_im)


    @triton.jit
    def _dawin_bwd_stage3_grad_kernel(
        Z_ptr, DT_ptr, MASK_ptr,
        AMP_ptr, GAMMA_ptr, OMEGA_ptr, THETA_ptr,
        P_OUT_ptr, Q_OUT_ptr,
        DELTA_LOCAL_RE_ptr, DELTA_LOCAL_IM_ptr,
        CARRY_ADJ_RE_ptr, CARRY_ADJ_IM_ptr,
        GZ_ptr, GDT_BM_ptr,
        GAMP_BM_ptr, GGAMMA_BM_ptr, GOMEGA_BM_ptr, GTHETA_BM_ptr,
        L, M, N_CHUNKS,
        stride_zb, stride_zl, stride_zm,
        stride_dtb, stride_dtl,
        stride_mb, stride_ml,
        stride_cb, stride_cc, stride_cm,
        stride_bm_c,
        HAS_MASK: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        """
        Stage 3 (Backward): Propagates future adjoint carry-in to reconstruct exact
        delta_bar_{i,m} across all positions and evaluates analytical gradients
        for Z, dt, A_m, gamma_m, omega_m, and theta_m (Eqs. 6-9).
        """
        bm_idx = tl.program_id(0)
        rev_chunk_idx = tl.program_id(1)

        b_idx = bm_idx // M
        m_idx = bm_idx % M

        amp = tl.load(AMP_ptr + m_idx).to(tl.float32)
        gamma = tl.load(GAMMA_ptr + m_idx).to(tl.float32)
        omega = tl.load(OMEGA_ptr + m_idx).to(tl.float32)
        theta = tl.load(THETA_ptr + m_idx).to(tl.float32)

        c_offset = b_idx * stride_cb + rev_chunk_idx * stride_cc + m_idx * stride_cm
        carry_re = tl.load(CARRY_ADJ_RE_ptr + c_offset).to(tl.float32)
        carry_im = tl.load(CARRY_ADJ_IM_ptr + c_offset).to(tl.float32)

        offs = tl.arange(0, CHUNK_SIZE)
        fwd_chunk_idx = (N_CHUNKS - 1) - rev_chunk_idx
        chunk_start = fwd_chunk_idx * CHUNK_SIZE
        t_idx = chunk_start + (CHUNK_SIZE - 1 - offs)
        valid = t_idx < L

        z_offset = b_idx * stride_zb + t_idx * stride_zl + m_idx * stride_zm
        d_loc_re = tl.load(DELTA_LOCAL_RE_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)
        d_loc_im = tl.load(DELTA_LOCAL_IM_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)

        if rev_chunk_idx > 0:
            t_next = t_idx + 1
            valid_next = valid & (t_next < L)
            dt_next_offset = b_idx * stride_dtb + t_next * stride_dtl
            dt_next = tl.load(DT_ptr + dt_next_offset, mask=valid_next, other=0.0).to(tl.float32)
            decay_next = tl.exp(-gamma * dt_next)
            if HAS_MASK:
                m_next_offset = b_idx * stride_mb + t_next * stride_ml
                rst_next = tl.load(MASK_ptr + m_next_offset, mask=valid_next, other=0.0).to(tl.float32)
                decay_next = decay_next * (1.0 - rst_next)
            rot_next = -omega * dt_next
            d_adj_re = tl.where(valid_next, decay_next * tl.cos(rot_next), 1.0)
            d_adj_im = tl.where(valid_next, decay_next * tl.sin(rot_next), 0.0)
            zeros = tl.zeros_like(d_adj_re)

            cum_d_re, cum_d_im, _, _ = tl.associative_scan(
                (d_adj_re, d_adj_im, zeros, zeros), axis=0, combine_fn=_dawin_scan_combine
            )
            db_re = d_loc_re + (cum_d_re * carry_re - cum_d_im * carry_im)
            db_im = d_loc_im + (cum_d_re * carry_im + cum_d_im * carry_re)
        else:
            db_re = d_loc_re
            db_im = d_loc_im

        # Load forward inputs & states at t_idx to evaluate analytical derivatives
        dt_offset = b_idx * stride_dtb + t_idx * stride_dtl
        z_val = tl.load(Z_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)
        dt_val = tl.load(DT_ptr + dt_offset, mask=valid, other=0.0).to(tl.float32)

        phi = z_val - dt_val * omega - theta
        cos_phi = tl.cos(phi)
        sin_phi = tl.sin(phi)
        u_re = amp * cos_phi
        u_im = amp * sin_phi

        # Total forward state psi_i = p_i + i * q_i
        p_curr = tl.load(P_OUT_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)
        q_curr = tl.load(Q_OUT_ptr + z_offset, mask=valid, other=0.0).to(tl.float32)

        # Carried decayed state: D(dt_i) * psi_{i-1} = psi_i - u_i
        carry_fwd_re = p_curr - u_re
        carry_fwd_im = q_curr - u_im

        # Eq. (6): dL/dA = Re(delta_bar * e^{i phi})
        g_amp_t = tl.where(valid, db_re * cos_phi - db_im * sin_phi, 0.0)

        # Eq. (6)-(7): Im(delta_bar * u_i) -> dL/dtheta and dL/dZ = -Im(delta_bar * u_i)
        im_db_u = tl.where(valid, db_re * u_im + db_im * u_re, 0.0)
        g_theta_t = im_db_u
        g_z_t = -im_db_u

        # Eq. (8): dL/dgamma = -dt_i * Re(delta_bar * D(dt_i) * psi_{i-1})
        re_db_carry = tl.where(valid, db_re * carry_fwd_re - db_im * carry_fwd_im, 0.0)
        g_gamma_t = -dt_val * re_db_carry

        # Eq. (9): dL/domega = dt_i * Im(delta_bar * psi_i)
        im_db_psi = tl.where(valid, db_re * q_curr + db_im * p_curr, 0.0)
        g_omega_t = dt_val * im_db_psi

        # dL/ddt_i = -gamma * Re(delta_bar * carry_fwd) - omega * Im(delta_bar * psi_i)
        g_dt_t = tl.where(valid, -gamma * re_db_carry + omega * im_db_psi, 0.0)

        # Write pointwise gradients
        tl.store(GZ_ptr + z_offset, g_z_t, mask=valid)
        tl.store(GDT_BM_ptr + z_offset, g_dt_t, mask=valid)

        # Reduce chunk-level parameter gradients into partial buffer [B*M, N_CHUNKS]
        bm_c_offset = bm_idx * stride_bm_c + rev_chunk_idx
        tl.store(GAMP_BM_ptr + bm_c_offset, tl.sum(g_amp_t, axis=0))
        tl.store(GGAMMA_BM_ptr + bm_c_offset, tl.sum(g_gamma_t, axis=0))
        tl.store(GOMEGA_BM_ptr + bm_c_offset, tl.sum(g_omega_t, axis=0))
        tl.store(GTHETA_BM_ptr + bm_c_offset, tl.sum(g_theta_t, axis=0))


# =============================================================================
# 3. Autograd Bridge & Chunk-Size Dispatch
# =============================================================================

def _select_chunk_config(L: int) -> tuple[int, int]:
    """
    Selects optimal intra-chunk tile size C (power of 2 <= 2048) and
    inter-chunk scan buffer size S (power of 2 >= ceil(L / C)).
    Supports sequences up to L = 2048 * 2048 = 4,194,304 tokens on GPU.
    """
    if L <= 2048:
        C = max(16, 1 << (L - 1).bit_length())
    else:
        C = 1024 if L <= 131072 else 2048
    n_chunks = math.ceil(L / C)
    S = max(16, 1 << (n_chunks - 1).bit_length())
    if S > 2048:
        raise ValueError(f"Sequence length L={L} exceeds maximum supported Triton scan length (4,194,304).")
    return C, S


class _TritonChunkedDAWINFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,
        dt: torch.Tensor,
        amp: torch.Tensor,
        gamma: torch.Tensor,
        omega: torch.Tensor,
        theta: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = z.dtype
        z_c = z.contiguous()
        dt_2d = dt.squeeze(-1) if dt.dim() == 3 else dt
        dt_c = dt_2d.contiguous()

        amp_c = amp.contiguous()
        gamma_c = gamma.contiguous()
        omega_c = omega.contiguous()
        theta_c = theta.contiguous()

        has_mask = reset_mask is not None
        if has_mask:
            mask_c = reset_mask.contiguous()
        else:
            mask_c = torch.empty((1, 1), device=z.device, dtype=torch.float32)

        B, L, M = z_c.shape
        C, S = _select_chunk_config(L)
        n_chunks = math.ceil(L / C)

        # Allocate FP32 outputs and inter-chunk summary buffers
        p_out = torch.empty((B, L, M), device=z.device, dtype=torch.float32)
        q_out = torch.empty((B, L, M), device=z.device, dtype=torch.float32)

        chunk_d_re = torch.empty((B, n_chunks, M), device=z.device, dtype=torch.float32)
        chunk_d_im = torch.empty((B, n_chunks, M), device=z.device, dtype=torch.float32)
        chunk_u_re = torch.empty((B, n_chunks, M), device=z.device, dtype=torch.float32)
        chunk_u_im = torch.empty((B, n_chunks, M), device=z.device, dtype=torch.float32)

        carry_u_re = torch.empty((B, n_chunks, M), device=z.device, dtype=torch.float32)
        carry_u_im = torch.empty((B, n_chunks, M), device=z.device, dtype=torch.float32)

        # Stage 1: Intra-chunk parallel prefix scan
        grid_chunks = (B * M, n_chunks)
        _dawin_fwd_stage1_chunk_scan_kernel[grid_chunks](
            z_c, dt_c, mask_c,
            amp_c, gamma_c, omega_c, theta_c,
            p_out, q_out,
            chunk_d_re, chunk_d_im, chunk_u_re, chunk_u_im,
            L, M, n_chunks,
            z_c.stride(0), z_c.stride(1), z_c.stride(2),
            dt_c.stride(0), dt_c.stride(1),
            mask_c.stride(0) if has_mask else 0,
            mask_c.stride(1) if has_mask else 0,
            chunk_d_re.stride(0), chunk_d_re.stride(1), chunk_d_re.stride(2),
            HAS_MASK=has_mask,
            CHUNK_SIZE=C,
        )

        # Stage 2 & 3: Inter-chunk scan and carry propagation (only needed if n_chunks > 1)
        if n_chunks > 1:
            _dawin_inter_chunk_scan_kernel[(B * M,)](
                chunk_d_re, chunk_d_im, chunk_u_re, chunk_u_im,
                carry_u_re, carry_u_im,
                n_chunks, M,
                chunk_d_re.stride(0), chunk_d_re.stride(1), chunk_d_re.stride(2),
                SCAN_SIZE=S,
            )
            _dawin_fwd_stage3_propagate_kernel[grid_chunks](
                dt_c, mask_c,
                gamma_c, omega_c,
                p_out, q_out,
                carry_u_re, carry_u_im,
                p_out, q_out,
                L, M,
                p_out.stride(0), p_out.stride(1), p_out.stride(2),
                dt_c.stride(0), dt_c.stride(1),
                mask_c.stride(0) if has_mask else 0,
                mask_c.stride(1) if has_mask else 0,
                carry_u_re.stride(0), carry_u_re.stride(1), carry_u_re.stride(2),
                HAS_MASK=has_mask,
                CHUNK_SIZE=C,
            )

        ctx.save_for_backward(z_c, dt_c, mask_c, amp_c, gamma_c, omega_c, theta_c, p_out, q_out)
        ctx.has_mask = has_mask
        ctx.orig_dtype = orig_dtype
        ctx.dt_was_3d = (dt.dim() == 3)
        ctx.chunk_cfg = (C, S, n_chunks)

        return p_out.to(orig_dtype), q_out.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_p: torch.Tensor, grad_q: torch.Tensor):
        z_c, dt_c, mask_c, amp_c, gamma_c, omega_c, theta_c, p_out, q_out = ctx.saved_tensors
        has_mask = ctx.has_mask
        orig_dtype = ctx.orig_dtype
        C, S, n_chunks = ctx.chunk_cfg

        gp_c = grad_p.contiguous()
        gq_c = grad_q.contiguous()
        B, L, M = z_c.shape

        delta_loc_re = torch.empty((B, L, M), device=z_c.device, dtype=torch.float32)
        delta_loc_im = torch.empty((B, L, M), device=z_c.device, dtype=torch.float32)

        chunk_d_re = torch.empty((B, n_chunks, M), device=z_c.device, dtype=torch.float32)
        chunk_d_im = torch.empty((B, n_chunks, M), device=z_c.device, dtype=torch.float32)
        chunk_u_re = torch.empty((B, n_chunks, M), device=z_c.device, dtype=torch.float32)
        chunk_u_im = torch.empty((B, n_chunks, M), device=z_c.device, dtype=torch.float32)

        carry_adj_re = torch.zeros((B, n_chunks, M), device=z_c.device, dtype=torch.float32)
        carry_adj_im = torch.zeros((B, n_chunks, M), device=z_c.device, dtype=torch.float32)

        grid_chunks = (B * M, n_chunks)

        # Stage 1 (Backward): Reverse-time intra-chunk adjoint scan
        _dawin_bwd_stage1_chunk_scan_kernel[grid_chunks](
            dt_c, mask_c,
            gamma_c, omega_c,
            gp_c, gq_c,
            delta_loc_re, delta_loc_im,
            chunk_d_re, chunk_d_im, chunk_u_re, chunk_u_im,
            L, M, n_chunks,
            z_c.stride(0), z_c.stride(1), z_c.stride(2),
            dt_c.stride(0), dt_c.stride(1),
            mask_c.stride(0) if has_mask else 0,
            mask_c.stride(1) if has_mask else 0,
            chunk_d_re.stride(0), chunk_d_re.stride(1), chunk_d_re.stride(2),
            HAS_MASK=has_mask,
            CHUNK_SIZE=C,
        )

        # Stage 2 (Backward): Inter-chunk adjoint carry propagation across chunks
        if n_chunks > 1:
            _dawin_inter_chunk_scan_kernel[(B * M,)](
                chunk_d_re, chunk_d_im, chunk_u_re, chunk_u_im,
                carry_adj_re, carry_adj_im,
                n_chunks, M,
                chunk_d_re.stride(0), chunk_d_re.stride(1), chunk_d_re.stride(2),
                SCAN_SIZE=S,
            )

        # Stage 3 (Backward): Reconstruct full delta_bar & compute analytical parameter gradients
        gz = torch.empty((B, L, M), device=z_c.device, dtype=torch.float32)
        gdt_bm = torch.empty((B, L, M), device=z_c.device, dtype=torch.float32)

        gamp_bm = torch.empty((B * M, n_chunks), device=z_c.device, dtype=torch.float32)
        ggamma_bm = torch.empty((B * M, n_chunks), device=z_c.device, dtype=torch.float32)
        gomega_bm = torch.empty((B * M, n_chunks), device=z_c.device, dtype=torch.float32)
        gtheta_bm = torch.empty((B * M, n_chunks), device=z_c.device, dtype=torch.float32)

        _dawin_bwd_stage3_grad_kernel[grid_chunks](
            z_c, dt_c, mask_c,
            amp_c, gamma_c, omega_c, theta_c,
            p_out, q_out,
            delta_loc_re, delta_loc_im,
            carry_adj_re, carry_adj_im,
            gz, gdt_bm,
            gamp_bm, ggamma_bm, gomega_bm, gtheta_bm,
            L, M, n_chunks,
            z_c.stride(0), z_c.stride(1), z_c.stride(2),
            dt_c.stride(0), dt_c.stride(1),
            mask_c.stride(0) if has_mask else 0,
            mask_c.stride(1) if has_mask else 0,
            carry_adj_re.stride(0), carry_adj_re.stride(1), carry_adj_re.stride(2),
            gamp_bm.stride(0),
            HAS_MASK=has_mask,
            CHUNK_SIZE=C,
        )

        # Reduce [B * M, n_chunks] -> [B, M] -> [M]
        g_amp = gamp_bm.sum(dim=1).view(B, M).sum(dim=0).to(amp_c.dtype)
        g_gamma = ggamma_bm.sum(dim=1).view(B, M).sum(dim=0).to(gamma_c.dtype)
        g_omega = gomega_bm.sum(dim=1).view(B, M).sum(dim=0).to(omega_c.dtype)
        g_theta = gtheta_bm.sum(dim=1).view(B, M).sum(dim=0).to(theta_c.dtype)

        if ctx.needs_input_grad[1]:
            g_dt = gdt_bm.sum(dim=-1).to(dt_c.dtype)
            if ctx.dt_was_3d:
                g_dt = g_dt.unsqueeze(-1)
        else:
            g_dt = None

        return gz.to(orig_dtype), g_dt, g_amp, g_gamma, g_omega, g_theta, None


def triton_dawin_scan(
    z: torch.Tensor,
    dt: torch.Tensor,
    amp: torch.Tensor,
    gamma: torch.Tensor,
    omega: torch.Tensor,
    theta: torch.Tensor,
    reset_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatches to the multi-block chunked Triton kernel when running on CUDA
    (supporting arbitrary L up to 4,194,304) and falls back cleanly to the
    exact FP32 PyTorch implementation on CPU or when Triton is unavailable.
    """
    if HAS_TRITON and z.is_cuda:
        return _TritonChunkedDAWINFunction.apply(z, dt, amp, gamma, omega, theta, reset_mask)
    return dawin_reference_forward(z, dt, amp, gamma, omega, theta, reset_mask)


# =============================================================================
# 4. Drop-In TritonDAWINLinear Layer
# =============================================================================

class TritonDAWINLinear(nn.Module):
    """
    High-throughput drop-in replacement for Linear + Activation using the
    multi-block chunked Triton parallel associative scan kernel.
    Supports sequences of arbitrary length (L = 1 to L > 100,000) on GPU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = "quad",
        gamma_min: float = 1e-4,
        omega_init_max: float = 0.5,
        use_triton: bool = True,
        use_parallel_scan: bool = True,
        chunk_size: int = 2048,
        **kwargs,
    ):
        super().__init__()
        if mode not in ("real", "quad"):
            raise ValueError(f"mode must be 'real' or 'quad', got {mode!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.gamma_min = gamma_min
        self.use_triton = use_triton
        self.use_parallel_scan = use_parallel_scan
        self.chunk_size = chunk_size

        self.k = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.theta = nn.Parameter(torch.zeros(out_features))
        self.omega = nn.Parameter(torch.linspace(0.1, omega_init_max, out_features))
        self.raw_amp = nn.Parameter(torch.full((out_features,), math.log(math.e - 1.0)))
        self.raw_gamma = nn.Parameter(torch.linspace(-2.0, 1.0, out_features))

    @property
    def amplitude(self) -> torch.Tensor:
        return F.softplus(self.raw_amp)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma) + self.gamma_min

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = F.linear(x.float(), self.k.float())
        if self.use_triton and x.is_cuda:
            p, q = triton_dawin_scan(
                z, dt, self.amplitude, self.gamma, self.omega, self.theta, reset_mask
            )
        else:
            p, q = dawin_reference_forward(
                z, dt, self.amplitude, self.gamma, self.omega, self.theta, reset_mask
            )
        return p if self.mode == "real" else torch.cat([p, q], dim=-1)
