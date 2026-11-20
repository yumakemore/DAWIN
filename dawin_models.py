import math
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dawin_core import DAWINLinear, _dawin_binary_op

try:
    from dawin_triton import TritonDAWINLinear, HAS_TRITON
except ImportError:
    TritonDAWINLinear = None
    HAS_TRITON = False

try:
    from torch._higher_order_ops import associative_scan
    HAS_ASSOC_SCAN = True
except ImportError:
    associative_scan = None
    HAS_ASSOC_SCAN = False

try:
    import torchcde
    HAS_TORCHCDE = True
except ImportError:
    torchcde = None
    HAS_TORCHCDE = False


def build_dawin_layer(
    in_features: int,
    out_features: int,
    mode: str = "quad",
    ablation: str = "none",
    use_parallel_scan: bool = True,
    use_triton: bool = True,
    gamma_min: float = 1e-4,
    omega_init_max: float = math.pi,
) -> nn.Module:
    """
    Factory returning TritonDAWINLinear when Triton is available and requested
    (with ablation == 'none'), falling back cleanly to PyTorch DAWINLinear on CPU
    or when ablation toggles are active.
    """
    if (
        use_triton
        and TritonDAWINLinear is not None
        and HAS_TRITON
        and ablation == "none"
    ):
        return TritonDAWINLinear(
            in_features=in_features,
            out_features=out_features,
            mode=mode,
            gamma_min=gamma_min,
            omega_init_max=omega_init_max,
            use_triton=True,
            use_parallel_scan=use_parallel_scan,
        )
    return DAWINLinear(
        in_features=in_features,
        out_features=out_features,
        mode=mode,
        ablation=ablation,
        use_parallel_scan=use_parallel_scan,
        gamma_min=gamma_min,
        omega_init_max=omega_init_max,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_matched_width(
    target_params: int,
    build_fn: Callable[[int], nn.Module],
    min_width: int = 8,
    max_width: int = 2048,
    max_rel_diff: float = 0.02,
) -> Tuple[int, int, float]:
    """
    Binary-search over hidden width `d` so that `build_fn(d)` matches
    `target_params` within `max_rel_diff` (default <= 2.0%).
    """
    best_w = min_width
    best_params = count_trainable_parameters(build_fn(min_width))
    best_rel = abs(best_params - target_params) / max(target_params, 1)

    lo, hi = min_width, max_width
    while lo <= hi:
        mid = (lo + hi) // 2
        m = build_fn(mid)
        p = count_trainable_parameters(m)
        rel = abs(p - target_params) / max(target_params, 1)
        if rel < best_rel:
            best_w, best_params, best_rel = mid, p, rel
        if p < target_params:
            lo = mid + 1
        elif p > target_params:
            hi = mid - 1
        else:
            break

    # Local fine-grain refinement around best_w
    for w in range(max(min_width, best_w - 6), min(max_width, best_w + 7)):
        m = build_fn(w)
        p = count_trainable_parameters(m)
        rel = abs(p - target_params) / max(target_params, 1)
        if rel < best_rel:
            best_w, best_params, best_rel = w, p, rel

    if best_rel > max_rel_diff:
        raise ValueError(
            f"Could not match target_params={target_params} within {max_rel_diff*100:.2f}% "
            f"(closest width={best_w} gave {best_params} params, diff={best_rel*100:.2f}%)."
        )
    return best_w, best_params, best_rel


# =====================================================================
# 1. DAWIN Drop-In Backbones (MLP, Transformer, Dynamic GNN)
# =====================================================================

class DAWINMLP(nn.Module):
    """
    Continuous-time Pre-Norm Residual MLP formed by replacing every feed-forward
    block with a LayerNorm -> DAWINLinear -> Linear residual block.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_width: int = 128,
        num_layers: int = 2,
        ablation: str = "none",
        use_parallel_scan: bool = True,
        use_triton: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_width = hidden_width
        self.num_layers = num_layers
        self.use_triton = use_triton

        d_model = 2 * hidden_width
        self.in_proj = nn.Linear(in_features, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.layers = nn.ModuleList([
            build_dawin_layer(
                in_features=d_model,
                out_features=hidden_width,
                mode="quad",
                ablation=ablation,
                use_parallel_scan=use_parallel_scan,
                use_triton=(use_triton and use_parallel_scan),
            )
            for _ in range(num_layers)
        ])
        self.out_projs = nn.ModuleList([nn.Linear(2 * hidden_width, d_model) for _ in range(num_layers)])
        self.readout = nn.Linear(d_model, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.in_proj(x)
        for norm, layer, proj in zip(self.norms, self.layers, self.out_projs):
            h = h + proj(layer(norm(h), dt, reset_mask=reset_mask))
        return self.readout(h)


class DAWINTransformerBlock(nn.Module):
    """
    Causal Transformer block whose static position-wise FFN is replaced by
    DAWINLinear to inject exact analytical continuous-time dynamics over dt.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dawin_width: int,
        dropout: float = 0.0,
        ablation: str = "none",
        use_parallel_scan: bool = True,
        use_triton: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dawin_ffn = build_dawin_layer(
            in_features=d_model,
            out_features=dawin_width,
            mode="quad",
            ablation=ablation,
            use_parallel_scan=use_parallel_scan,
            use_triton=(use_triton and use_parallel_scan),
        )
        self.proj_out = nn.Linear(2 * dawin_width, d_model)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        attn_mask: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        h2 = self.norm2(x)
        w_out = self.dawin_ffn(h2, dt, reset_mask=reset_mask)
        return x + self.proj_out(w_out)


class DAWINTransformer(nn.Module):
    """
    Causal sequence Transformer with DAWIN-based continuous-time FFN blocks.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dawin_width: Optional[int] = None,
        ablation: str = "none",
        use_parallel_scan: bool = True,
        use_triton: bool = True,
    ):
        super().__init__()
        if dawin_width is None:
            dawin_width = d_model
        self.in_proj = nn.Linear(in_features, d_model)
        self.blocks = nn.ModuleList(
            [
                DAWINTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dawin_width=dawin_width,
                    ablation=ablation,
                    use_parallel_scan=use_parallel_scan,
                    use_triton=(use_triton and use_parallel_scan),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        L = x.size(1)
        attn_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, dt, attn_mask=attn_mask, reset_mask=reset_mask)
        return self.readout(self.norm_f(h))


class DAWINGNN(nn.Module):
    """
    Continuous-time Dynamic Graph Neural Network where asynchronous node updates
    across inter-event intervals dt are governed by stacked DAWINLinear units.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_width: int = 96,
        num_layers: int = 2,
        ablation: str = "none",
        use_parallel_scan: bool = True,
        use_triton: bool = True,
    ):
        super().__init__()
        self.msg_encoder = nn.Linear(in_features, hidden_width)
        layers = []
        d_in = hidden_width
        for _ in range(num_layers):
            layers.append(
                build_dawin_layer(
                    in_features=d_in,
                    out_features=hidden_width,
                    mode="quad",
                    ablation=ablation,
                    use_parallel_scan=use_parallel_scan,
                    use_triton=(use_triton and use_parallel_scan),
                )
            )
            d_in = 2 * hidden_width
        self.node_evolvers = nn.ModuleList(layers)
        self.readout = nn.Linear(d_in, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_in] or [B, L, N, d_in]
            dt: [B, L] or [B, L, N]
            adj: Optional normalized adjacency [N, N] or [B, N, N]
        """
        if x.dim() == 4:
            B, L, N, D = x.shape
            msg = self.msg_encoder(x)
            if adj is not None:
                if adj.dim() == 2:
                    msg = torch.einsum("ij,bljd->blid", adj, msg)
                else:
                    msg = torch.einsum("bij,bljd->blid", adj, msg)
            h = msg.permute(0, 2, 1, 3).reshape(B * N, L, -1)
            dt_flat = dt.permute(0, 2, 1).reshape(B * N, L)
            rm_flat = (
                reset_mask.permute(0, 2, 1).reshape(B * N, L)
                if reset_mask is not None
                else None
            )
            for layer in self.node_evolvers:
                h = layer(h, dt_flat, reset_mask=rm_flat)
            out = self.readout(h).reshape(B, N, L, -1).permute(0, 2, 1, 3)
            return out

        h = self.msg_encoder(x)
        for layer in self.node_evolvers:
            h = layer(h, dt, reset_mask=reset_mask)
        return self.readout(h)


# =====================================================================
# 2. Parameter-Matched Conventional & Neural ODE/CDE/SSM Comparators
# =====================================================================

class MatchedReLUMLP(nn.Module):
    """Parameter-matched static ReLU MLP receiving [x, dt]."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_width: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        d_in = in_features + 1
        for _ in range(num_layers):
            layers.append(nn.Linear(d_in, hidden_width))
            layers.append(nn.ReLU())
            d_in = hidden_width
        self.net = nn.Sequential(*layers)
        self.readout = nn.Linear(d_in, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dt.dim() == 2:
            dt = dt.unsqueeze(-1)
        inp = torch.cat([x, dt], dim=-1)
        return self.readout(self.net(inp))


class MatchedTimeTransformer(nn.Module):
    """Parameter-matched causal Transformer receiving [x, dt, log(1+dt)]."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_features + 2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.readout = nn.Linear(d_model, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dt.dim() == 2:
            dt = dt.unsqueeze(-1)
        L = x.size(1)
        attn_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        inp = torch.cat([x, dt, torch.log1p(dt.clamp_min(0.0))], dim=-1)
        h = self.in_proj(inp)
        h = self.encoder(h, mask=attn_mask)
        return self.readout(h)


class _CDEFunc(nn.Module):
    def __init__(self, hidden_channels: int, path_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.path_channels = path_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels * path_channels),
            nn.Tanh(),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        return self.net(z).view(B, self.hidden_channels, self.path_channels)


class MatchedNeuralCDE(nn.Module):
    """
    Native Neural Controlled Differential Equation [Kidger et al., NeurIPS 2020]
    over cumulative timestamp paths X(tau) = [tau, x]. Uses torchcde when
    installed, with a piecewise-linear RK4 solver fallback.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_width: int = 64,
    ):
        super().__init__()
        self.path_channels = in_features + 1
        self.hidden_width = hidden_width
        self.init_proj = nn.Linear(self.path_channels, hidden_width)
        self.func = _CDEFunc(hidden_width, self.path_channels)
        self.readout = nn.Linear(hidden_width, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dt.dim() == 3:
            dt = dt.squeeze(-1)
        B, L, _ = x.shape
        tau = torch.cumsum(dt.clamp_min(0.0), dim=1)  # [B, L]
        path = torch.cat([tau.unsqueeze(-1), x], dim=-1)  # [B, L, D+1]

        if HAS_TORCHCDE and L > 1:
            t_grid = torch.arange(L, device=x.device, dtype=x.dtype)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
                path, t=t_grid
            )
            spline = torchcde.CubicSpline(coeffs, t=t_grid)
            z0 = self.init_proj(spline.evaluate(t_grid[0]))
            z_traj = torchcde.cdeint(
                X=spline,
                func=self.func,
                z0=z0,
                t=t_grid,
                method="rk4",
            )  # [B, L, H]
            return self.readout(z_traj)

        # Pure PyTorch piecewise-linear RK4 fallback
        z = self.init_proj(path[:, 0, :])
        outs = [self.readout(z)]
        for i in range(1, L):
            dX = path[:, i, :] - path[:, i - 1, :]  # [B, C]
            # 2-substep RK4 along linear path segment
            for _ in range(2):
                dx_sub = 0.5 * dX
                k1 = torch.bmm(self.func(tau[:, i], z), dx_sub.unsqueeze(-1)).squeeze(-1)
                k2 = torch.bmm(self.func(tau[:, i], z + 0.5 * k1), dx_sub.unsqueeze(-1)).squeeze(-1)
                k3 = torch.bmm(self.func(tau[:, i], z + 0.5 * k2), dx_sub.unsqueeze(-1)).squeeze(-1)
                k4 = torch.bmm(self.func(tau[:, i], z + k3), dx_sub.unsqueeze(-1)).squeeze(-1)
                z = z + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            outs.append(self.readout(z))
        return torch.stack(outs, dim=1)


class _S5Layer(nn.Module):
    """
    Continuous-time Simplified State Space (S5) layer [Smith et al., ICLR 2023]
    with complex diagonal eigenvalues Lambda = -gamma + i*omega, ZOH
    discretization over irregular dt, and parallel associative prefix scan.
    """
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.raw_gamma = nn.Parameter(torch.linspace(-2.0, 1.0, state_dim))
        self.omega = nn.Parameter(torch.linspace(0.1, 2.0 * math.pi, state_dim))
        self.B_re = nn.Parameter(torch.randn(state_dim, d_model) / math.sqrt(d_model))
        self.B_im = nn.Parameter(torch.randn(state_dim, d_model) / math.sqrt(d_model))
        self.C_re = nn.Parameter(torch.randn(d_model, state_dim) / math.sqrt(state_dim))
        self.C_im = nn.Parameter(torch.randn(d_model, state_dim) / math.sqrt(state_dim))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dt.dim() == 2:
            dt = dt.unsqueeze(-1)
        B, L, _ = x.shape
        gamma = (F.softplus(self.raw_gamma) + 1e-4).view(1, 1, -1)
        omega = self.omega.view(1, 1, -1)

        # Discrete transition A_bar = exp((-gamma + i*omega) * dt)
        decay = torch.exp(-gamma * dt)
        if reset_mask is not None:
            decay = decay * (1.0 - reset_mask.unsqueeze(-1).to(decay.dtype))
        angle = omega * dt
        d_re = decay * torch.cos(angle)
        d_im = decay * torch.sin(angle)

        # ZOH input scaling: (A_bar - I) / Lambda * B * x
        lam_re = -gamma
        lam_im = omega
        denom = (lam_re * lam_re + lam_im * lam_im).clamp_min(1e-8)
        num_re = d_re - 1.0
        num_im = d_im
        zoh_re = (num_re * lam_re + num_im * lam_im) / denom
        zoh_im = (num_im * lam_re - num_re * lam_im) / denom

        bx_re = F.linear(x, self.B_re)
        bx_im = F.linear(x, self.B_im)
        u_re = zoh_re * bx_re - zoh_im * bx_im
        u_im = zoh_re * bx_im + zoh_im * bx_re

        if HAS_ASSOC_SCAN and L > 1:
            _, _, h_re, h_im = associative_scan(
                _dawin_binary_op, (d_re, d_im, u_re, u_im), dim=1, combine_mode="generic"
            )
        else:
            p_list, q_list = [], []
            p_prev = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
            q_prev = torch.zeros_like(p_prev)
            for t in range(L):
                p_curr = d_re[:, t] * p_prev - d_im[:, t] * q_prev + u_re[:, t]
                q_curr = d_re[:, t] * q_prev + d_im[:, t] * p_prev + u_im[:, t]
                p_list.append(p_curr)
                q_list.append(q_curr)
                p_prev, q_prev = p_curr, q_curr
            h_re = torch.stack(p_list, dim=1)
            h_im = torch.stack(q_list, dim=1)

        y = F.linear(h_re, self.C_re) - F.linear(h_im, self.C_im) + x * self.D
        return F.gelu(y)


class MatchedS5(nn.Module):
    """Parameter-matched multi-layer continuous-time S5 comparator."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int = 96,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.layers = nn.ModuleList(
            [_S5Layer(d_model=d_model, state_dim=d_model) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.readout = nn.Linear(d_model, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h), dt, reset_mask=reset_mask)
        return self.readout(h)


class _MambaBlock(nn.Module):
    """
    Continuous-time Selective State Space (Mamba) block [Gu & Dao, COLM 2024]
    modulated by physical elapsed intervals dt and evaluated via parallel scan.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=2,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner + 1, self.d_inner + 2 * d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dt.dim() == 2:
            dt = dt.unsqueeze(-1)
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Causal depthwise conv
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective parameters conditioned on [x_conv, dt]
        sel = self.x_proj(torch.cat([x_conv, dt], dim=-1))
        dt_rank, B_sel, C_sel = torch.split(
            sel, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_rank)) * dt.clamp_min(1e-4)  # [B, L, d_inner]

        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        # Discretize: A_bar = exp(delta * A) -> [B, L, d_inner, d_state]
        d_re = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state))
        if reset_mask is not None:
            d_re = d_re * (1.0 - reset_mask.view(B, L, 1, 1).to(d_re.dtype))
        d_im = torch.zeros_like(d_re)

        u_re = (delta * x_conv).unsqueeze(-1) * B_sel.unsqueeze(2)
        u_im = torch.zeros_like(u_re)

        if HAS_ASSOC_SCAN and L > 1:
            _, _, h_re, _ = associative_scan(
                _dawin_binary_op, (d_re, d_im, u_re, u_im), dim=1, combine_mode="generic"
            )
        else:
            h_list = []
            h_prev = torch.zeros(
                B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
            )
            for t in range(L):
                h_curr = d_re[:, t] * h_prev + u_re[:, t]
                h_list.append(h_curr)
                h_prev = h_curr
            h_re = torch.stack(h_list, dim=1)

        y = (h_re * C_sel.unsqueeze(2)).sum(dim=-1) + x_conv * self.D
        y = y * F.silu(z_branch)
        return self.out_proj(y)


class MatchedMamba(nn.Module):
    """Parameter-matched multi-layer continuous-time Mamba comparator."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int = 96,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.layers = nn.ModuleList(
            [_MambaBlock(d_model=d_model) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.readout = nn.Linear(d_model, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h), dt, reset_mask=reset_mask)
        return self.readout(h)


class MatchedGRUGNN(nn.Module):
    """Parameter-matched dynamic GNN comparator using Time-GRU node updates."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_width: int = 96,
        num_layers: int = 2,
    ):
        super().__init__()
        self.msg_encoder = nn.Linear(in_features + 1, hidden_width)
        self.gru = nn.GRU(
            input_size=hidden_width,
            hidden_size=hidden_width,
            num_layers=num_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_width, out_features)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 4:
            B, L, N, D = x.shape
            dt_exp = dt.unsqueeze(-1) if dt.dim() == 3 else dt
            msg = self.msg_encoder(torch.cat([x, dt_exp], dim=-1))
            if adj is not None:
                if adj.dim() == 2:
                    msg = torch.einsum("ij,bljd->blid", adj, msg)
                else:
                    msg = torch.einsum("bij,bljd->blid", adj, msg)
            h = msg.permute(0, 2, 1, 3).reshape(B * N, L, -1)
            out, _ = self.gru(h)
            return self.readout(out).reshape(B, N, L, -1).permute(0, 2, 1, 3)

        if dt.dim() == 2:
            dt = dt.unsqueeze(-1)
        h = self.msg_encoder(torch.cat([x, dt], dim=-1))
        out, _ = self.gru(h)
        return self.readout(out)
