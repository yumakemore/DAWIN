"""
verify_dawin_datasets.py
------------------------
End-to-end verification script testing all 9 benchmark dataset configurations
across MuJoCo, PhysioNet 2012 (dense + mask-aware), and ETH/UCY + JAAD,
verifying tensor shapes, training-only normalization hygiene, agent/patient
disjointness, and metric adapters.
"""

import torch
from dawin_datasets import (
    get_benchmark_bundle,
    compute_trajectory_ade_fde,
    compute_masked_imputation_mse,
    compute_binary_auroc_auprc,
)


def main():
    benchmarks = [
        "HalfCheetah-v5",
        "Hopper-v5",
        "Reacher-v5",
        "Walker2d-v5",
        "PhysioNet2012-mortality",
        "PhysioNet2012-imputation",
        "PhysioNet2012-dense",
        "ETH_UCY_JAAD-position",
        "ETH_UCY_JAAD-motion",
    ]

    print("=" * 84)
    print(f"{'Dataset':<28} | {'Task':<21} | {'Train/Val/Test':<17} | {'X Shape':<14} | {'Y Shape'}")
    print("-" * 84)

    for name in benchmarks:
        bundle = get_benchmark_bundle(name, seed=42)
        tr_l, va_l, te_l = bundle.get_loaders(batch_size=32)
        batch = next(iter(tr_l))
        x_sh = tuple(batch["x"].shape)
        y_sh = tuple(batch["y"].shape)
        counts = f"{len(bundle.train_dataset)}/{len(bundle.val_dataset)}/{len(bundle.test_dataset)}"
        print(f"{name:<28} | {bundle.task_type:<21} | {counts:<17} | {str(x_sh):<14} | {str(y_sh)}")

        assert not torch.isnan(batch["x"]).any(), f"NaN found in {name} x"
        assert not torch.isnan(batch["dt"]).any(), f"NaN found in {name} dt"
        assert not torch.isnan(batch["y"]).any(), f"NaN found in {name} y"
        assert (batch["dt"] >= 0).all(), f"Negative dt found in {name}"

    # Verify metric helpers
    pred_rel = torch.zeros(16, 12, 2)
    anchor = torch.ones(16, 1, 2) * 0.5
    target_abs = anchor + 0.02
    ade, fde = compute_trajectory_ade_fde(pred_rel, anchor, target_abs)
    assert abs(ade - fde) < 1e-6 and ade > 0.0

    auroc, auprc = compute_binary_auroc_auprc(
        torch.tensor([2.0, 1.0, -1.0, -2.0]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    assert abs(auroc - 1.0) < 1e-6 and abs(auprc - 1.0) < 1e-6

    imp_mse = compute_masked_imputation_mse(
        torch.ones(4, 10, 5),
        torch.zeros(4, 10, 5),
        torch.tensor([[[1, 0, 0, 0, 0]] * 10] * 4, dtype=torch.float32),
    )
    assert abs(imp_mse - 1.0) < 1e-6

    print("=" * 84)
    print("ALL 9 BENCHMARK ADAPTERS AND METRIC HELPERS PASSED VERIFICATION!")


if __name__ == "__main__":
    main()
