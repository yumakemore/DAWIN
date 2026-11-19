"""
dawin_datasets.py
-----------------
Unified dataset loaders, preprocessing pipelines, and task-specific adapters for
evaluating Damped Atomic Wave Integration Neurons (DAWIN) across:
  1. MuJoCo Irregular Physical Dynamics (HalfCheetah-v5, Hopper-v5, Reacher-v5, Walker2d-v5)
     + Damped Oscillator & dt-Irrelevant Control [Yu & Kim, 2026b (WSP)]
  2. PhysioNet Challenge 2012 (Set A):
     - Mode A: Dense 10-channel state-displacement protocol [WSP]
     - Mode B: Mask-aware clinical mortality (AUROC/AUPRC) & imputation (MSE) [Neural CDE / ContiFormer]
  3. ETH/UCY + JAAD Pedestrian Trajectories:
     - Agent-disjoint (source, agent_id) partitioning, FPS=2.5, T_obs=8, T_pred=12
     - Motion-feature (6D) & Position-only (2D) modes [Yu & Kim, 2026a (WFB)]

All datasets expose standardized `(x, dt, y)` or `(x, dt, mask, y)` tensors
compatible with `dawin_models.py` and `run_dawin_experiments.py`, plus built-in
synthetic/offline fallbacks when external raw data files or gymnasium are absent.
"""

from __future__ import annotations

import os
import glob
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DatasetBundle:
    """Container holding train/val/test datasets, statistics, and task metadata."""
    name: str
    task_type: Literal["regression", "binary_classification", "imputation", "trajectory"]
    in_features: int
    out_features: int
    seq_len: int
    pred_len: int = 1
    train_dataset: Optional[Dataset] = None
    val_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    stats: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def get_loaders(
        self,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader


class IrregularSequenceDataset(Dataset):
    """
    Standardized dataset for irregular-time sequence modeling.
    Items return a dictionary with:
      - 'x': [L, d_in] float32 spatial/feature inputs
      - 'dt': [L] float32 non-negative elapsed intervals
      - 'y': [L, d_out] or [d_out] float32 targets
      - 'mask': [L, d_obs] float32 binary observation mask (1=observed, 0=missing)
      - 'raw_target': optional unnormalized target for physical/coordinate metrics (ADE/FDE)
    """

    def __init__(
        self,
        x: np.ndarray,
        dt: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray] = None,
        raw_target: Optional[np.ndarray] = None,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.dt = torch.as_tensor(dt, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.mask = (
            torch.as_tensor(mask, dtype=torch.float32)
            if mask is not None
            else torch.ones_like(self.x)
        )
        self.raw_target = (
            torch.as_tensor(raw_target, dtype=torch.float32)
            if raw_target is not None
            else None
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "x": self.x[idx],
            "dt": self.dt[idx],
            "y": self.y[idx],
            "mask": self.mask[idx],
        }
        if self.raw_target is not None:
            item["raw_target"] = self.raw_target[idx]
        return item


def _compute_train_stats(arr: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Compute feature-wise mean and safe std from training split only."""
    flat = arr.reshape(-1, arr.shape[-1])
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean, std


def _extract_noncrossing_windows(
    episodes: List[Dict[str, np.ndarray]],
    window_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fixed-length windows strictly within episode boundaries
    (never crossing episodes; incomplete tails excluded, matching WSP Sec. 5.1).
    """
    xs, dts, ys, masks = [], [], [], []
    for ep in episodes:
        T = ep["x"].shape[0]
        if T < window_len:
            continue
        for start in range(0, T - window_len + 1, stride):
            end = start + window_len
            xs.append(ep["x"][start:end])
            dts.append(ep["dt"][start:end])
            ys.append(ep["y"][start:end])
            masks.append(ep.get("mask", np.ones_like(ep["x"]))[start:end])
    if not xs:
        raise ValueError(f"No episodes had length >= window_len={window_len}.")
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.stack(dts, axis=0).astype(np.float32),
        np.stack(ys, axis=0).astype(np.float32),
        np.stack(masks, axis=0).astype(np.float32),
    )


# =============================================================================
# 1. MUJOCO IRREGULAR PHYSICAL DYNAMICS ADAPTER (WSP Protocol)
# =============================================================================

MUJOCO_ENV_SPECS = {
    "HalfCheetah-v5": {"state_dim": 17, "action_dim": 6, "h": 0.05, "window_len": 32, "train_stride": 16, "eval_stride": 32},
    "Hopper-v5":      {"state_dim": 11, "action_dim": 3, "h": 0.008, "window_len": 4, "train_stride": 2, "eval_stride": 4},
    "Reacher-v5":     {"state_dim": 10, "action_dim": 2, "h": 0.02, "window_len": 32, "train_stride": 16, "eval_stride": 32},
    "Walker2d-v5":    {"state_dim": 17, "action_dim": 6, "h": 0.008, "window_len": 32, "train_stride": 16, "eval_stride": 32},
}


def _simulate_mujoco_episodes(
    env_name: str,
    episode_seeds: List[int],
    num_transitions: int,
    step_choices: Tuple[int, ...],
    rng: np.random.Generator,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate irregularly subsampled MuJoCo episodes where a held action a_i
    evolves the simulator over n_i physics steps (dt_i = n_i * h),
    predicting standardized next-state displacement ds_i = s_{i+1} - s_i.
    Uses `gymnasium` when installed, or an analytical coupled multibody surrogate
    preserving exact dimensions and step multiples when offline.
    """
    spec = MUJOCO_ENV_SPECS[env_name]
    s_dim, a_dim, h = spec["state_dim"], spec["action_dim"], spec["h"]
    episodes = []

    try:
        import gymnasium as gym
        has_gym = True
    except ImportError:
        has_gym = False

    if has_gym:
        env = gym.make(env_name)
        for ep_seed in episode_seeds:
            obs, _ = env.reset(seed=int(ep_seed))
            ep_rng = np.random.default_rng(ep_seed)
            xs, dts, ys = [], [], []
            for _ in range(num_transitions):
                n_steps = int(ep_rng.choice(step_choices))
                action = ep_rng.uniform(-1.0, 1.0, size=(a_dim,)).astype(np.float32)
                s_curr = np.asarray(obs, dtype=np.float32)
                terminated = False
                for _ in range(n_steps):
                    obs, _, term, trunc, _ = env.step(action)
                    if term or trunc:
                        terminated = True
                        break
                s_next = np.asarray(obs, dtype=np.float32)
                dt_val = n_steps * h
                xs.append(np.concatenate([s_curr, action], axis=-1))
                dts.append(dt_val)
                ys.append(s_next - s_curr)
                if terminated:
                    break
            if len(xs) > 0:
                episodes.append({
                    "x": np.asarray(xs, dtype=np.float32),
                    "dt": np.asarray(dts, dtype=np.float32),
                    "y": np.asarray(ys, dtype=np.float32),
                })
        env.close()
    else:
        # Coupled nonlinear articulated-body surrogate matching exact dimensions & dt scaling
        A_sys = rng.normal(0.0, 0.35, size=(s_dim, s_dim)).astype(np.float32)
        U, _, Vt = np.linalg.svd(A_sys)
        A_sys = (U @ np.diag(np.linspace(-1.8, -0.15, s_dim)) @ Vt).astype(np.float32)
        Omega_sys = np.diag(np.linspace(1.5, 9.0, s_dim)).astype(np.float32)
        B_act = rng.normal(0.0, 0.6, size=(s_dim, a_dim)).astype(np.float32)

        for ep_seed in episode_seeds:
            ep_rng = np.random.default_rng(ep_seed)
            s_curr = ep_rng.normal(0.0, 0.5, size=(s_dim,)).astype(np.float32)
            xs, dts, ys = [], [], []
            for _ in range(num_transitions):
                n_steps = int(ep_rng.choice(step_choices))
                dt_val = float(n_steps * h)
                action = ep_rng.uniform(-1.0, 1.0, size=(a_dim,)).astype(np.float32)
                # Exact sub-step integration over n_steps
                s_tmp = s_curr.copy()
                for _ in range(n_steps):
                    ds = (A_sys @ s_tmp + np.sin(Omega_sys @ s_tmp) * 0.25 + B_act @ action) * h
                    s_tmp = s_tmp + ds
                xs.append(np.concatenate([s_curr, action], axis=-1))
                dts.append(dt_val)
                ys.append(s_tmp - s_curr)
                s_curr = s_tmp
            episodes.append({
                "x": np.asarray(xs, dtype=np.float32),
                "dt": np.asarray(dts, dtype=np.float32),
                "y": np.asarray(ys, dtype=np.float32),
            })

    return episodes


def load_mujoco_irregular_bundle(
    env_name: str = "HalfCheetah-v5",
    data_dir: Optional[str] = None,
    n_train_episodes: int = 80,
    n_val_episodes: int = 16,
    n_test_episodes: int = 16,
    transitions_per_episode: int = 160,
    train_step_choices: Tuple[int, ...] = (1, 2, 3, 4, 5),
    test_step_choices: Tuple[int, ...] = (1, 2, 3, 5, 7),
    seed: int = 42,
) -> DatasetBundle:
    """
    Loads or generates irregularly subsampled MuJoCo dynamics with:
      - Disjoint episode seeds across train / val / test
      - Out-of-range test intervals (test_step_choices includes step=7 > train max=5)
      - Training-only z-score normalization of inputs, dt, and target displacements ds
    """
    assert env_name in MUJOCO_ENV_SPECS, f"Unsupported MuJoCo env: {env_name}"
    spec = MUJOCO_ENV_SPECS[env_name]
    rng = np.random.default_rng(seed)

    # Disjoint episode seeds
    train_seeds = list(range(1000, 1000 + n_train_episodes))
    val_seeds = list(range(5000, 5000 + n_val_episodes))
    test_seeds = list(range(9000, 9000 + n_test_episodes))

    if data_dir and os.path.exists(os.path.join(data_dir, f"{env_name}_splits.npz")):
        loaded = np.load(os.path.join(data_dir, f"{env_name}_splits.npz"))
        x_tr, dt_tr, y_tr = loaded["x_tr"], loaded["dt_tr"], loaded["y_tr"]
        x_va, dt_va, y_va = loaded["x_va"], loaded["dt_va"], loaded["y_va"]
        x_te, dt_te, y_te = loaded["x_te"], loaded["dt_te"], loaded["y_te"]
    else:
        tr_eps = _simulate_mujoco_episodes(env_name, train_seeds, transitions_per_episode, train_step_choices, rng)
        va_eps = _simulate_mujoco_episodes(env_name, val_seeds, transitions_per_episode, train_step_choices, rng)
        te_eps = _simulate_mujoco_episodes(env_name, test_seeds, transitions_per_episode, test_step_choices, rng)

        x_tr, dt_tr, y_tr, _ = _extract_noncrossing_windows(tr_eps, spec["window_len"], spec["train_stride"])
        x_va, dt_va, y_va, _ = _extract_noncrossing_windows(va_eps, spec["window_len"], spec["eval_stride"])
        x_te, dt_te, y_te, _ = _extract_noncrossing_windows(te_eps, spec["window_len"], spec["eval_stride"])

    # Compute normalization statistics strictly on training partition
    x_mean, x_std = _compute_train_stats(x_tr)
    y_mean, y_std = _compute_train_stats(y_tr)
    dt_scale = float(np.max(dt_tr)) if np.max(dt_tr) > 0 else 1.0

    def norm_split(x, dt, y):
        x_n = (x - x_mean) / x_std
        y_n = (y - y_mean) / y_std
        dt_n = dt / dt_scale
        return x_n, dt_n, y_n

    x_tr_n, dt_tr_n, y_tr_n = norm_split(x_tr, dt_tr, y_tr)
    x_va_n, dt_va_n, y_va_n = norm_split(x_va, dt_va, y_va)
    x_te_n, dt_te_n, y_te_n = norm_split(x_te, dt_te, y_te)

    return DatasetBundle(
        name=env_name,
        task_type="regression",
        in_features=spec["state_dim"] + spec["action_dim"],
        out_features=spec["state_dim"],
        seq_len=spec["window_len"],
        train_dataset=IrregularSequenceDataset(x_tr_n, dt_tr_n, y_tr_n, raw_target=y_tr),
        val_dataset=IrregularSequenceDataset(x_va_n, dt_va_n, y_va_n, raw_target=y_va),
        test_dataset=IrregularSequenceDataset(x_te_n, dt_te_n, y_te_n, raw_target=y_te),
        stats={
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "dt_scale": np.array([dt_scale], dtype=np.float32),
        },
        meta={"physics_h": spec["h"], "window_len": spec["window_len"]},
    )


# =============================================================================
# 2. PHYSIONET CHALLENGE 2012 ADAPTER (Dense WSP Mode + Mask-Aware Mode)
# =============================================================================

PHYSIONET_10_CHANNELS = [
    "HR", "NISysABP", "NIDiasABP", "NIMAP", "RespRate",
    "Temp", "SpO2", "GCS", "Urine", "Lactate"
]

PHYSIONET_35_CHANNELS = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP",
    "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate",
    "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine"
]


def _parse_physionet_set_a(
    raw_dir: str,
    channels: List[str],
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """
    Parses raw PhysioNet 2012 Set A patient `.txt` files and `Outcomes-a.txt`.
    Aligns measurements at the union of observation timestamps (in seconds),
    recording exact binary observation masks M_{i,c} in {0, 1}.
    """
    outcomes_path = os.path.join(raw_dir, "Outcomes-a.txt")
    outcomes_map: Dict[str, int] = {}
    if os.path.exists(outcomes_path):
        with open(outcomes_path, "r") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    outcomes_map[parts[0].strip()] = int(parts[5].strip())

    ch_to_idx = {c: i for i, c in enumerate(channels)}
    D = len(channels)
    patient_records = []
    labels = []

    txt_files = sorted(glob.glob(os.path.join(raw_dir, "set-a", "*.txt")))
    if not txt_files:
        txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))

    for fp in txt_files:
        base = os.path.basename(fp)
        if base.startswith("Outcomes"):
            continue
        rec_id = os.path.splitext(base)[0]
        time_events: Dict[float, Dict[int, float]] = {}

        with open(fp, "r") as f:
            _ = f.readline()  # Time,Parameter,Value
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue
                t_str, param, val_str = parts
                if param not in ch_to_idx:
                    continue
                try:
                    hh, mm = map(int, t_str.split(":"))
                    t_sec = float(hh * 3600 + mm * 60)
                    val = float(val_str)
                except ValueError:
                    continue
                if np.isnan(val):
                    continue
                time_events.setdefault(t_sec, {})[ch_to_idx[param]] = val

        sorted_times = sorted(time_events.keys())
        if len(sorted_times) < 4:
            continue

        T = len(sorted_times)
        vals = np.full((T, D), np.nan, dtype=np.float32)
        mask = np.zeros((T, D), dtype=np.float32)
        times = np.asarray(sorted_times, dtype=np.float32)

        for idx_t, t_sec in enumerate(sorted_times):
            for c_idx, val in time_events[t_sec].items():
                vals[idx_t, c_idx] = val
                mask[idx_t, c_idx] = 1.0

        dt = np.zeros(T, dtype=np.float32)
        dt[1:] = np.maximum(times[1:] - times[:-1], 0.0)

        patient_records.append({"vals": vals, "mask": mask, "dt": dt, "times": times})
        labels.append(outcomes_map.get(rec_id, 0))

    return patient_records, np.asarray(labels, dtype=np.float32)


def _generate_synthetic_physionet(
    n_patients: int,
    n_channels: int,
    seq_len: int,
    seed: int = 42,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """
    Offline realistic surrogate for PhysioNet Challenge 2012 Set A with irregular
    ICU sampling intervals (in seconds), channel-specific sparsity masks (~75% missing),
    and physiological deterioration dynamics correlated with mortality.
    """
    rng = np.random.default_rng(seed)
    records = []
    labels = np.zeros(n_patients, dtype=np.float32)

    for p in range(n_patients):
        mortality = int(rng.random() < 0.14)
        labels[p] = float(mortality)
        T = seq_len + int(rng.integers(4, 24))
        # Irregular ICU intervals in seconds (e.g., 5 min to 4 hours)
        dt = rng.exponential(scale=1800.0, size=(T,)).astype(np.float32)
        dt[0] = 0.0
        times = np.cumsum(dt)

        # Latent physiological trajectories
        freqs = rng.uniform(0.5, 3.0, size=(n_channels,)).astype(np.float32)
        phases = rng.uniform(0.0, 2.0 * math.pi, size=(n_channels,)).astype(np.float32)
        t_hours = (times / 3600.0)[:, None]
        drift = (0.04 * mortality) * t_hours * rng.normal(1.0, 0.3, size=(1, n_channels))
        latent = np.sin(2.0 * math.pi * freqs[None, :] * (t_hours / 24.0) + phases[None, :]) + drift
        latent += rng.normal(0.0, 0.15, size=(T, n_channels))

        # Sparse observation mask (vital signs observed ~40%, labs ~15%)
        obs_prob = np.linspace(0.45, 0.15, n_channels)[None, :]
        mask = (rng.random((T, n_channels)) < obs_prob).astype(np.float32)
        mask[0, : max(2, n_channels // 4)] = 1.0

        vals = np.where(mask > 0.5, latent, np.nan).astype(np.float32)
        records.append({"vals": vals, "mask": mask, "dt": dt, "times": times})

    return records, labels


def load_physionet2012_bundle(
    raw_dir: Optional[str] = None,
    protocol: Literal["mask_aware_mortality", "mask_aware_imputation", "dense_wsp"] = "mask_aware_mortality",
    seq_len: int = 48,
    seed: int = 42,
) -> DatasetBundle:
    """
    Loads PhysioNet Challenge 2012 (Set A) with patient-disjoint 70/15/15 splits.
    Supports three protocols:
      1. 'mask_aware_mortality': Concatenates [carried_values, observation_mask, time_since_last_obs]
         (3 * D channels, D=35) and predicts binary in-hospital mortality (AUROC / AUPRC).
      2. 'mask_aware_imputation': Masks 20% of observed measurements at evaluation time and
         predicts standardized values at masked coordinates (Imputation MSE / NRMSE).
      3. 'dense_wsp': Replicates the 10-channel forward-filled dense next-step displacement
         protocol from WSP [Yu & Kim, 2026b] (L=32, stride=16/32) for direct comparison.
    """
    channels = PHYSIONET_10_CHANNELS if protocol == "dense_wsp" else PHYSIONET_35_CHANNELS
    D = len(channels)

    if raw_dir and os.path.exists(raw_dir):
        records, labels = _parse_physionet_set_a(raw_dir, channels)
    else:
        default_len = 64 if protocol == "dense_wsp" else seq_len
        records, labels = _generate_synthetic_physionet(
            n_patients=400, n_channels=D, seq_len=default_len, seed=seed
        )

    rng = np.random.default_rng(seed)
    n_total = len(records)
    perm = rng.permutation(n_total)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    tr_idx = perm[:n_train]
    va_idx = perm[n_train : n_train + n_val]
    te_idx = perm[n_train + n_val :]

    # Compute training-cohort channel medians for initial missing imputation
    tr_vals_concat = np.concatenate([records[i]["vals"] for i in tr_idx], axis=0)
    tr_medians = np.nanmedian(tr_vals_concat, axis=0).astype(np.float32)
    tr_medians = np.where(np.isnan(tr_medians), 0.0, tr_medians)

    def forward_fill_patient(vals: np.ndarray, mask: np.ndarray, dt: np.ndarray):
        T, C = vals.shape
        filled = np.zeros((T, C), dtype=np.float32)
        delta_obs = np.zeros((T, C), dtype=np.float32)
        last_val = tr_medians.copy()
        elapsed = np.zeros(C, dtype=np.float32)

        for t in range(T):
            if t > 0:
                elapsed += dt[t]
            obs_c = mask[t] > 0.5
            last_val = np.where(obs_c, vals[t], last_val)
            elapsed = np.where(obs_c, 0.0, elapsed)
            filled[t] = last_val
            delta_obs[t] = elapsed
        return filled, delta_obs

    # Fit z-score mean/std on training filled values
    tr_filled_list = [forward_fill_patient(records[i]["vals"], records[i]["mask"], records[i]["dt"])[0] for i in tr_idx]
    tr_filled_all = np.concatenate(tr_filled_list, axis=0)
    ch_mean = np.mean(tr_filled_all, axis=0).astype(np.float32)
    ch_std = np.where(np.std(tr_filled_all, axis=0) < 1e-6, 1.0, np.std(tr_filled_all, axis=0)).astype(np.float32)

    if protocol == "dense_wsp":
        # WSP protocol: L=32 windows, stride=16 (train) / 32 (eval), predict ds = s_{t+1} - s_t
        win_len = 32
        def build_dense_windows(indices, stride):
            eps = []
            for idx in indices:
                rec = records[idx]
                filled, _ = forward_fill_patient(rec["vals"], rec["mask"], rec["dt"])
                filled_norm = (filled - ch_mean) / ch_std
                if filled_norm.shape[0] <= win_len:
                    continue
                x_seq = filled_norm[:-1]
                dt_seq = rec["dt"][1:] / 3600.0  # hours
                y_seq = filled_norm[1:] - filled_norm[:-1]
                eps.append({"x": x_seq, "dt": dt_seq, "y": y_seq})
            return _extract_noncrossing_windows(eps, win_len, stride)

        x_tr, dt_tr, y_tr, _ = build_dense_windows(tr_idx, stride=16)
        x_va, dt_va, y_va, _ = build_dense_windows(va_idx, stride=32)
        x_te, dt_te, y_te, _ = build_dense_windows(te_idx, stride=32)

        y_mean, y_std = _compute_train_stats(y_tr)
        y_tr = (y_tr - y_mean) / y_std
        y_va = (y_va - y_mean) / y_std
        y_te = (y_te - y_mean) / y_std

        return DatasetBundle(
            name="PhysioNet2012-DenseWSP",
            task_type="regression",
            in_features=D,
            out_features=D,
            seq_len=win_len,
            train_dataset=IrregularSequenceDataset(x_tr, dt_tr, y_tr),
            val_dataset=IrregularSequenceDataset(x_va, dt_va, y_va),
            test_dataset=IrregularSequenceDataset(x_te, dt_te, y_te),
            stats={"ch_mean": ch_mean, "ch_std": ch_std, "y_mean": y_mean, "y_std": y_std},
        )

    # Mask-aware protocols (fixed sequence budget L=seq_len, padded/truncated from admission)
    def build_mask_aware_split(indices, holdout_prob: float = 0.0):
        xs, dts, ys, masks = [], [], [], []
        split_rng = np.random.default_rng(seed + len(indices))
        for idx in indices:
            rec = records[idx]
            vals = rec["vals"].copy()
            obs_mask = rec["mask"].copy()
            dt_hours = rec["dt"] / 3600.0

            target_mask = obs_mask.copy()
            if protocol == "mask_aware_imputation" and holdout_prob > 0.0:
                drop = (split_rng.random(obs_mask.shape) < holdout_prob) & (obs_mask > 0.5)
                target_mask = drop.astype(np.float32)
                obs_mask = np.where(drop, 0.0, obs_mask)
                vals = np.where(drop, np.nan, vals)

            filled, delta_obs = forward_fill_patient(vals, obs_mask, dt_hours)
            filled_norm = (filled - ch_mean) / ch_std
            # Concatenate [normalized_values, binary_obs_mask, log1p(time_since_last_obs)]
            feat = np.concatenate([filled_norm, obs_mask, np.log1p(delta_obs)], axis=-1)

            T = feat.shape[0]
            if T >= seq_len:
                feat_pad = feat[:seq_len]
                dt_pad = dt_hours[:seq_len]
                tmask_pad = target_mask[:seq_len]
                true_vals_pad = ((np.where(np.isnan(rec["vals"][:seq_len]), filled[:seq_len], rec["vals"][:seq_len]) - ch_mean) / ch_std)
            else:
                pad_len = seq_len - T
                feat_pad = np.pad(feat, ((0, pad_len), (0, 0)), mode="constant")
                dt_pad = np.pad(dt_hours, (0, pad_len), mode="constant")
                tmask_pad = np.pad(target_mask, ((0, pad_len), (0, 0)), mode="constant")
                true_raw = np.where(np.isnan(rec["vals"]), filled, rec["vals"])
                true_vals_pad = np.pad((true_raw - ch_mean) / ch_std, ((0, pad_len), (0, 0)), mode="constant")

            xs.append(feat_pad)
            dts.append(dt_pad)
            masks.append(tmask_pad)
            if protocol == "mask_aware_mortality":
                ys.append(np.array([labels[idx]], dtype=np.float32))
            else:
                ys.append(true_vals_pad)

        return (
            np.asarray(xs, dtype=np.float32),
            np.asarray(dts, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(masks, dtype=np.float32),
        )

    holdout_p = 0.20 if protocol == "mask_aware_imputation" else 0.0
    x_tr, dt_tr, y_tr, m_tr = build_mask_aware_split(tr_idx, holdout_prob=holdout_p)
    x_va, dt_va, y_va, m_va = build_mask_aware_split(va_idx, holdout_prob=holdout_p)
    x_te, dt_te, y_te, m_te = build_mask_aware_split(te_idx, holdout_prob=holdout_p)

    task_type = "binary_classification" if protocol == "mask_aware_mortality" else "imputation"
    out_dim = 1 if protocol == "mask_aware_mortality" else D

    return DatasetBundle(
        name=f"PhysioNet2012-{protocol}",
        task_type=task_type,
        in_features=3 * D,
        out_features=out_dim,
        seq_len=seq_len,
        train_dataset=IrregularSequenceDataset(x_tr, dt_tr, y_tr, mask=m_tr),
        val_dataset=IrregularSequenceDataset(x_va, dt_va, y_va, mask=m_va),
        test_dataset=IrregularSequenceDataset(x_te, dt_te, y_te, mask=m_te),
        stats={"ch_mean": ch_mean, "ch_std": ch_std},
        meta={"protocol": protocol, "num_channels": D},
    )


# =============================================================================
# 3. ETH/UCY + JAAD PEDESTRIAN TRAJECTORIES ADAPTER (WFB Protocol)
# =============================================================================

def _generate_synthetic_pedestrian_trajectories(
    n_agents: int = 360,
    fps: float = 2.5,
    drop_prob: float = 0.35,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    Offline realistic surrogate matching the ETH/UCY + JAAD irregular trajectory
    statistics from WFB [Yu & Kim, 2026a]:
      - FPS = 2.5, missed detections creating irregular frame gaps dt_i = (F_i - F_{i-1}) / FPS
      - Normalized bounding-box / world coordinates (x, y) with smooth nonlinear turns
      - Distinct `(source, agent_id)` keys to guarantee zero agent leakage across splits
    """
    rng = np.random.default_rng(seed)
    sources = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2", "JAAD"]
    trajectories = []

    for idx in range(n_agents):
        src = sources[idx % len(sources)]
        agent_id = f"{src}_agent_{idx:04d}"
        total_frames = int(rng.integers(32, 65))

        # Simulate irregular missed detections (keeping frame 0)
        keep = rng.random(total_frames) > drop_prob
        keep[0] = True
        frame_indices = np.nonzero(keep)[0]
        if len(frame_indices) < 22:
            frame_indices = np.arange(0, 44, 2)

        t_sec = frame_indices.astype(np.float32) / fps
        dt = np.zeros_like(t_sec)
        dt[1:] = (frame_indices[1:] - frame_indices[:-1]).astype(np.float32) / fps

        # Smooth curvilinear pedestrian motion in normalized [0, 1] coordinates
        x0, y0 = rng.uniform(0.15, 0.85, size=2)
        speed = rng.uniform(0.015, 0.045)
        heading0 = rng.uniform(0.0, 2.0 * math.pi)
        turn_rate = rng.uniform(-0.18, 0.18)

        headings = heading0 + turn_rate * t_sec
        vx_true = speed * np.cos(headings)
        vy_true = speed * np.sin(headings)
        pos_x = x0 + np.cumsum(vx_true * np.where(dt > 0, dt, 1.0 / fps))
        pos_y = y0 + np.cumsum(vy_true * np.where(dt > 0, dt, 1.0 / fps))

        # Finite-difference kinematic features matching WFB Eq. (30)-(31)
        vx = np.zeros_like(pos_x)
        vy = np.zeros_like(pos_y)
        ax = np.zeros_like(pos_x)
        ay = np.zeros_like(pos_y)

        safe_dt = np.where(dt[1:] > 1e-6, dt[1:], 1.0 / fps)
        vx[1:] = (pos_x[1:] - pos_x[:-1]) / safe_dt
        vy[1:] = (pos_y[1:] - pos_y[:-1]) / safe_dt
        ax[1:] = (vx[1:] - vx[:-1]) / safe_dt
        ay[1:] = (vy[1:] - vy[:-1]) / safe_dt

        trajectories.append({
            "source": src,
            "agent_id": agent_id,
            "pos": np.stack([pos_x, pos_y], axis=-1).astype(np.float32),
            "motion": np.stack([pos_x, pos_y, vx, vy, ax, ay], axis=-1).astype(np.float32),
            "dt": dt.astype(np.float32),
        })

    return trajectories


def load_eth_ucy_jaad_bundle(
    npz_path: Optional[str] = None,
    feature_mode: Literal["position_only", "motion_features"] = "position_only",
    t_obs: int = 8,
    t_pred: int = 12,
    fps: float = 2.5,
    seed: int = 42,
) -> DatasetBundle:
    """
    Loads ETH/UCY + JAAD irregular pedestrian trajectories following the exact
    protocol of WFB [Yu & Kim, 2026a]:
      - Splits strictly by unique `(source, agent_id)` pair (70% train / 15% val / 15% test)
        before normalization and sliding-window extraction so no agent overlaps splits.
      - Input window: T_obs = 8 observations; Target: next T_pred = 12 (x, y) coordinates.
      - Supports 'position_only' (2D: [x, y], eliminating velocity/acceleration leakage)
        and 'motion_features' (6D: [x, y, vx, vy, ax, ay]).
    """
    if npz_path and os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        trajectories = list(data["trajectories"])
    else:
        trajectories = _generate_synthetic_pedestrian_trajectories(
            n_agents=420, fps=fps, drop_prob=0.35, seed=seed
        )

    # Group by unique (source, agent_id)
    unique_keys = sorted({(tr["source"], tr["agent_id"]) for tr in trajectories})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_keys))

    n_tr = int(0.70 * len(unique_keys))
    n_va = int(0.15 * len(unique_keys))
    tr_keys = {unique_keys[i] for i in perm[:n_tr]}
    va_keys = {unique_keys[i] for i in perm[n_tr : n_tr + n_va]}
    te_keys = {unique_keys[i] for i in perm[n_tr + n_va :]}

    feat_key = "pos" if feature_mode == "position_only" else "motion"
    total_win = t_obs + t_pred

    def extract_windows_for_keys(key_set):
        xs, dts, ys_raw = [], [], []
        for tr in trajectories:
            if (tr["source"], tr["agent_id"]) not in key_set:
                continue
            feat = tr[feat_key]
            pos = tr["pos"]
            dt = tr["dt"]
            T = feat.shape[0]
            if T < total_win:
                continue
            for s in range(0, T - total_win + 1):
                xs.append(feat[s : s + t_obs])
                dts.append(dt[s : s + t_obs])
                ys_raw.append(pos[s + t_obs : s + total_win])
        return (
            np.asarray(xs, dtype=np.float32),
            np.asarray(dts, dtype=np.float32),
            np.asarray(ys_raw, dtype=np.float32),
        )

    x_tr, dt_tr, y_tr_raw = extract_windows_for_keys(tr_keys)
    x_va, dt_va, y_va_raw = extract_windows_for_keys(va_keys)
    x_te, dt_te, y_te_raw = extract_windows_for_keys(te_keys)

    # Standardize inputs and dt using training statistics only
    x_mean, x_std = _compute_train_stats(x_tr)
    dt_mean = float(np.mean(dt_tr))
    dt_std = float(np.std(dt_tr)) if np.std(dt_tr) > 1e-6 else 1.0

    # Relative displacement target anchored at last observed position p_{t_obs - 1}
    # (while storing absolute coordinates in raw_target for exact ADE/FDE computation)
    def prepare_split(x, dt, y_abs):
        x_norm = (x - x_mean) / x_std
        dt_norm = np.maximum((dt - dt_mean) / dt_std + 1.0, 0.0)
        anchor = y_abs[:, :1, :] * 0.0 + x[:, -1:, :2]  # [N, 1, 2]
        y_rel = y_abs - anchor  # [N, T_pred, 2]
        return x_norm, dt_norm, y_rel

    x_tr_n, dt_tr_n, y_tr_rel = prepare_split(x_tr, dt_tr, y_tr_raw)
    x_va_n, dt_va_n, y_va_rel = prepare_split(x_va, dt_va, y_va_raw)
    x_te_n, dt_te_n, y_te_rel = prepare_split(x_te, dt_te, y_te_raw)

    in_dim = 2 if feature_mode == "position_only" else 6

    return DatasetBundle(
        name=f"ETH_UCY_JAAD-{feature_mode}",
        task_type="trajectory",
        in_features=in_dim,
        out_features=t_pred * 2,
        seq_len=t_obs,
        pred_len=t_pred,
        train_dataset=IrregularSequenceDataset(x_tr_n, dt_tr_n, y_tr_rel, raw_target=y_tr_raw),
        val_dataset=IrregularSequenceDataset(x_va_n, dt_va_n, y_va_rel, raw_target=y_va_raw),
        test_dataset=IrregularSequenceDataset(x_te_n, dt_te_n, y_te_rel, raw_target=y_te_raw),
        stats={
            "x_mean": x_mean,
            "x_std": x_std,
            "dt_mean": np.array([dt_mean], dtype=np.float32),
            "dt_std": np.array([dt_std], dtype=np.float32),
        },
        meta={"t_obs": t_obs, "t_pred": t_pred, "fps": fps, "feature_mode": feature_mode},
    )


# =============================================================================
# 4. UNIFIED BENCHMARK REGISTRY & METRIC ADAPTERS
# =============================================================================

def get_benchmark_bundle(
    dataset_name: str,
    data_root: Optional[str] = None,
    seed: int = 42,
) -> DatasetBundle:
    """
    Unified factory resolving any dataset referenced in DAWIN Section 5:
      - 'HalfCheetah-v5', 'Hopper-v5', 'Reacher-v5', 'Walker2d-v5'
      - 'PhysioNet2012-mortality', 'PhysioNet2012-imputation', 'PhysioNet2012-dense'
      - 'ETH_UCY_JAAD-position', 'ETH_UCY_JAAD-motion'
    """
    if dataset_name in MUJOCO_ENV_SPECS:
        return load_mujoco_irregular_bundle(env_name=dataset_name, data_dir=data_root, seed=seed)
    elif dataset_name == "PhysioNet2012-mortality":
        return load_physionet2012_bundle(raw_dir=data_root, protocol="mask_aware_mortality", seed=seed)
    elif dataset_name == "PhysioNet2012-imputation":
        return load_physionet2012_bundle(raw_dir=data_root, protocol="mask_aware_imputation", seed=seed)
    elif dataset_name == "PhysioNet2012-dense":
        return load_physionet2012_bundle(raw_dir=data_root, protocol="dense_wsp", seed=seed)
    elif dataset_name == "ETH_UCY_JAAD-position":
        return load_eth_ucy_jaad_bundle(npz_path=data_root, feature_mode="position_only", seed=seed)
    elif dataset_name == "ETH_UCY_JAAD-motion":
        return load_eth_ucy_jaad_bundle(npz_path=data_root, feature_mode="motion_features", seed=seed)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


def compute_trajectory_ade_fde(
    pred_rel: torch.Tensor,
    anchor_pos: torch.Tensor,
    target_abs: torch.Tensor,
) -> Tuple[float, float]:
    """
    Computes Average Displacement Error (ADE) and Final Displacement Error (FDE)
    matching WFB Eqs. (33)-(34).
    Args:
        pred_rel: [B, T_pred, 2] predicted relative displacement from anchor
        anchor_pos: [B, 1, 2] unnormalized last observed (x, y) position
        target_abs: [B, T_pred, 2] unnormalized ground-truth future coordinates
    """
    pred_abs = pred_rel + anchor_pos
    l2_dist = torch.norm(pred_abs - target_abs, p=2, dim=-1)  # [B, T_pred]
    ade = float(l2_dist.mean().item())
    fde = float(l2_dist[:, -1].mean().item())
    return ade, fde


def compute_masked_imputation_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eval_mask: torch.Tensor,
) -> float:
    """Computes MSE strictly over held-out observed coordinates (eval_mask == 1)."""
    denom = eval_mask.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * eval_mask).sum() / denom
    return float(mse.item())


def compute_binary_auroc_auprc(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float]:
    """Pure-PyTorch/NumPy exact AUROC and AUPRC computation (no sklearn dependency required)."""
    probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
    y_true = targets.view(-1).detach().cpu().numpy()

    order = np.argsort(-probs)
    y_sorted = y_true[order]
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5, float(n_pos / max(len(y_true), 1))

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    auroc = float(np.trapezoid(tpr, fpr))

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_pos
    recall_diff = np.diff(np.concatenate([[0.0], recall]))
    auprc = float(np.sum(precision * recall_diff))
    return auroc, auprc
