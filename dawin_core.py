import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch._higher_order_ops import associative_scan
    HAS_ASSOC_SCAN = True
except ImportError:
    HAS_ASSOC_SCAN = False


def _dawin_binary_op(
    elem_l: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    elem_r: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Associative binary operator for paired-real complex tuples:
    (d_r, u_r) (+) (d_l, u_l) = (d_r * d_l, d_r * u_l + u_r)
    where d = d_re + i * d_im and u = u_re + i * u_im.
    """
    d_re_l, d_im_l, u_re_l, u_im_l = elem_l
    d_re_r, d_im_r, u_re_r, u_im_r = elem_r

    # Complex multiplication: d_new = d_r * d_l
    d_re = d_re_r * d_re_l - d_im_r * d_im_l
    d_im = d_re_r * d_im_l + d_im_r * d_re_l

    # Complex multiply-add: u_new = d_r * u_l + u_r
    u_re = (d_re_r * u_re_l - d_im_r * u_im_l) + u_re_r
    u_im = (d_re_r * u_im_l + d_im_r * u_re_l) + u_im_r
    return (d_re, d_im, u_re, u_im)


class DAWINLinear(nn.Module):
    """
    Damped Atomic Wave Integration Neuron (DAWIN) Layer.

    Drop-in replacement for Linear + Activation blocks across MLPs, Transformers,
    and GNNs. Supports exact parallel associative scan, serial recurrence fallback,
    and built-in component ablation modes:
      - 'none': Full DAWIN neuron (atomic spatiotemporal phase + damped integration)
      - 'undamped': Removes damping (gamma = 0)
      - 'memoryless': Instantaneous damping (gamma -> inf, recovers WFB 1.0)
      - 'cosine_only': Zeroes imaginary quadrature propagation between steps
      - 'decoupled': Removes spatial wavevector k^T x from phase (linear injection SSM control)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = "quad",
        gamma_min: float = 1e-4,
        omega_init_max: float = 2.0 * math.pi,
        use_parallel_scan: bool = True,
        ablation: str = "none",
    ):
        super().__init__()
        if mode not in ("real", "quad"):
            raise ValueError(f"mode must be 'real' or 'quad', got {mode!r}")
        if ablation not in ("none", "undamped", "memoryless", "cosine_only", "decoupled"):
            raise ValueError(f"Unsupported ablation mode: {ablation!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.gamma_min = gamma_min
        self.use_parallel_scan = use_parallel_scan and HAS_ASSOC_SCAN
        self.ablation = ablation

        # Learnable atomic wave & damper parameters
        self.k = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(max(1, in_features)))
        self.theta = nn.Parameter(torch.zeros(out_features))
        self.omega = nn.Parameter(torch.linspace(0.1, omega_init_max, out_features))
        self.raw_amp = nn.Parameter(torch.full((out_features,), math.log(math.e - 1.0)))
        self.raw_gamma = nn.Parameter(torch.linspace(-2.0, 1.0, out_features))

        if ablation == "decoupled":
            self.linear_inj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter("linear_inj", None)

    @property
    def amplitude(self) -> torch.Tensor:
        return F.softplus(self.raw_amp)

    @property
    def gamma(self) -> torch.Tensor:
        if self.ablation == "undamped":
            return torch.zeros_like(self.raw_gamma)
        return F.softplus(self.raw_gamma) + self.gamma_min

    def laplacian_penalty(self) -> torch.Tensor:
        k = getattr(self, "k", getattr(self, "wavevector", getattr(self, "wavenumber", None)))
        if k is None:
            k = next(p for p in self.parameters() if p.ndim == 2)
        k2 = k.float().pow(2).sum(dim=-1)
        if hasattr(self, "amp"):
            a = self.amp.float() if isinstance(self.amp, torch.Tensor) else self.amp().float()
        elif hasattr(self, "raw_amp"):
            a = torch.nn.functional.softplus(self.raw_amp.float())
        else:
            a = torch.ones_like(k2)
        return (a.pow(2) * k2.pow(2)).mean()

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Spatial input tensor of shape [B, L, d_in]
            dt: Non-negative elapsed intervals of shape [B, L] or [B, L, 1]
            reset_mask: Optional binary reset mask [B, L] (1 resets carried state)
        Returns:
            Output tensor of shape [B, L, M] (mode='real') or [B, L, 2M] (mode='quad')
        """
        B, L, _ = x.shape
        if dt is None:
            dt = torch.zeros(B, L, 1, device=x.device, dtype=x.dtype)
        elif dt.dim() == 2:
            dt = dt.unsqueeze(-1)

        # Retain FP32 precision inside wave phase & recurrence scan
        orig_dtype = x.dtype
        x_f = x.float()
        dt_f = dt.float().clamp(min=0.0)

        amp = self.amplitude.float().view(1, 1, -1)
        gamma = self.gamma.float().view(1, 1, -1)
        omega = self.omega.float().view(1, 1, -1)
        theta = self.theta.float().view(1, 1, -1)

        if self.ablation == "decoupled":
            phi = -dt_f * omega - theta
            inj_scale = F.linear(x_f, self.linear_inj.weight.float())
            u_re = inj_scale * amp * torch.cos(phi)
            u_im = inj_scale * amp * torch.sin(phi)
        else:
            phi = F.linear(x_f, self.k.float()) - dt_f * omega - theta
            u_re = amp * torch.cos(phi)
            u_im = amp * torch.sin(phi)

        if self.ablation == "memoryless":
            p = u_re
            q = torch.zeros_like(u_im) if self.ablation == "cosine_only" else u_im
            out = p if self.mode == "real" else torch.cat([p, q], dim=-1)
            return out.to(orig_dtype)

        decay = torch.exp(-gamma * dt_f)
        if reset_mask is not None:
            decay = decay * (1.0 - reset_mask.unsqueeze(-1).float())

        rot_angle = -omega * dt_f
        d_re = decay * torch.cos(rot_angle)
        d_im = decay * torch.sin(rot_angle)

        if self.ablation == "cosine_only":
            # Ablate quadrature coupling: propagate only real scalar state
            p_list = []
            p_prev = torch.zeros(B, self.out_features, device=x.device, dtype=torch.float32)
            for t in range(L):
                p_curr = d_re[:, t] * p_prev + u_re[:, t]
                p_list.append(p_curr)
                p_prev = p_curr
            p = torch.stack(p_list, dim=1)
            q = torch.zeros_like(p)
        elif self.use_parallel_scan and L > 1:
            _, _, p, q = associative_scan(
                _dawin_binary_op, (d_re, d_im, u_re, u_im), dim=1, combine_mode="generic"
            )
        else:
            p_list, q_list = [], []
            p_prev = torch.zeros(B, self.out_features, device=x.device, dtype=torch.float32)
            q_prev = torch.zeros_like(p_prev)
            for t in range(L):
                p_curr = d_re[:, t] * p_prev - d_im[:, t] * q_prev + u_re[:, t]
                q_curr = d_re[:, t] * q_prev + d_im[:, t] * p_prev + u_im[:, t]
                p_list.append(p_curr)
                q_list.append(q_curr)
                p_prev, q_prev = p_curr, q_curr
            p = torch.stack(p_list, dim=1)
            q = torch.stack(q_list, dim=1)

        out = p if self.mode == "real" else torch.cat([p, q], dim=-1)
        return out.to(orig_dtype)


def compute_dawin_laplacian_loss(model: nn.Module) -> torch.Tensor:
    pens = [mod.laplacian_penalty() for mod in model.modules() if isinstance(mod, DAWINLinear)]
    return torch.stack(pens).mean() if pens else torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
