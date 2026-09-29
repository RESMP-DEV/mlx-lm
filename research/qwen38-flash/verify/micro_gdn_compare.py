#!/usr/bin/env python3
"""Stage-by-stage comparison of reference vs MLX GDN micro dumps."""

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ref = np.load(HERE / "micro-ref.npz")
mlx = np.load(HERE / "micro-mlx.npz")


def d(name, r, m):
    r, m = r.astype(np.float64), m.astype(np.float64)
    if r.shape != m.shape:
        print(f"{name:18s} SHAPE {r.shape} vs {m.shape}")
        return
    diff = np.abs(r - m)
    rel = diff.max() / (np.abs(r).max() + 1e-30)
    print(f"{name:18s} max|d|={diff.max():.3e}  rel={rel:.2e}  |ref|max={np.abs(r).max():.3e}")


# Raw post-conv tensors. Reference q/k are repeat_interleaved to Hv heads.
rep = ref["kd_q"].shape[2] // mlx["q_raw"].shape[2]
d("q_raw", ref["kd_q"][:, :, 0::rep], mlx["q_raw"])
d("k_raw", ref["kd_k"][:, :, 0::rep], mlx["k_raw"])
d("v", ref["kd_v"], mlx["v"])
d("g", ref["kd_g"], mlx["g"])
d("beta", ref["kd_beta"], mlx["beta"])

# Normalization check: reference kernel normalized q/k internally with
# dk**-0.5 on q after l2norm. Reconstruct the reference-normalized q/k
# from the raw reference tensors and compare against MLX's.
q_ref = ref["kd_q"].astype(np.float64)
k_ref = ref["kd_k"].astype(np.float64)
inv = q_ref.shape[-1] ** -0.5


def l2(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


qn_ref = inv * l2(q_ref)
kn_ref = l2(k_ref)
d("q_norm_vs_ref", qn_ref[:, :, 0::rep], mlx["q_norm_port"])
d("k_norm_vs_ref", kn_ref[:, :, 0::rep], mlx["k_norm_port"])
d("q_norm_fromref", qn_ref[:, :, 0::rep], mlx["q_norm_fromref"])

# Core kernel outputs.
d("core_port", ref["kd_core"], mlx["core_port"])
d("core_fromref", ref["kd_core"], mlx["core_fromref"])

d("gdn_out", ref["gdn_out"], mlx["gdn_out"])

# Per-position profile of the reference-input kernel comparison.
c = np.abs(ref["kd_core"].astype(np.float64) - mlx["core_fromref"].astype(np.float64))
print("core_fromref per-position max|d|:",
      " ".join(f"[{i}]={c.max(axis=(0, 2, 3))[i]:.1e}" for i in range(c.shape[1])))
