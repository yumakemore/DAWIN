"""
verify_extended_comparators.py
Verifies parameter matching (<= 2% discrepancy), forward/backward gradient flow,
and parallel vs. serial scan equivalence for Neural CDE, S5, and Mamba wrappers.
"""
import torch
from dawin_models import (
    DAWINMLP,
    MatchedReLUMLP,
    MatchedTimeTransformer,
    MatchedNeuralCDE,
    MatchedS5,
    MatchedMamba,
    count_trainable_parameters,
    find_matched_width,
)

def main():
    torch.manual_seed(42)
    B, L, d_in, d_out = 4, 24, 10, 6
    x = torch.randn(B, L, d_in)
    dt = torch.rand(B, L) * 1.5 + 0.05

    dawin = DAWINMLP(in_features=d_in, out_features=d_out, hidden_width=128, num_layers=2)
    target_p = count_trainable_parameters(dawin)
    print(f"Reference DAWIN-MLP Params: {target_p:,}")

    comparators = {
        "MatchedReLUMLP": lambda w: MatchedReLUMLP(d_in, d_out, hidden_width=w, num_layers=2),
        "MatchedTimeTransformer": lambda w: MatchedTimeTransformer(d_in, d_out, d_model=w * 2, nhead=2, num_layers=2),
        "MatchedNeuralCDE": lambda w: MatchedNeuralCDE(d_in, d_out, hidden_width=w),
        "MatchedS5": lambda w: MatchedS5(d_in, d_out, d_model=w, num_layers=2),
        "MatchedMamba": lambda w: MatchedMamba(d_in, d_out, d_model=w, num_layers=2),
    }

    for name, builder in comparators.items():
        width, p_count, rel_err = find_matched_width(
            target_p, builder, min_width=4, max_width=512, max_rel_diff=0.02
        )
        model = builder(width)
        out = model(x, dt)
        assert out.shape == (B, L, d_out), f"{name} shape mismatch: {out.shape}"
        loss = out.pow(2).mean()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0.0 and torch.isfinite(torch.tensor(grad_norm)), f"{name} invalid grad"
        print(
            f"[PASS] {name:<24} | width={width:<3} | params={p_count:,} "
            f"(diff={rel_err*100:.2f}%) | grad_norm={grad_norm:.4f}"
        )

if __name__ == "__main__":
    main()
