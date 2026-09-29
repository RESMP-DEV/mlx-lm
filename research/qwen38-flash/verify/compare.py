#!/usr/bin/env python3
"""Compare reference and MLX logits dumped by the parity harness."""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    overlaps = []
    for ra, rb in zip(a, b):
        ta = set(np.argsort(-ra)[:k].tolist())
        tb = set(np.argsort(-rb)[:k].tolist())
        overlaps.append(len(ta & tb) / k)
    return float(np.mean(overlaps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=2e-4)
    ap.add_argument("--name", default="tiny-qwen4exp")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--mlx", default=None)
    args = ap.parse_args()
    ref_path = args.ref or str(HERE / f"{args.name}-ref.npz")
    mlx_path = args.mlx or str(HERE / f"{args.name}-mlx.npz")

    ref = np.load(ref_path)
    mlx = np.load(mlx_path)
    ok = True
    for name in ref.files:
        r, m = ref[name], mlx[name]
        if r.shape != m.shape:
            print(f"{name}: SHAPE MISMATCH {r.shape} vs {m.shape}")
            ok = False
            continue
        diff = np.abs(r.astype(np.float64) - m.astype(np.float64))
        argmax_agree = int((r.argmax(-1) == m.argmax(-1)).sum())
        n = r.shape[0]
        overlap = topk_overlap(r, m)
        passed = diff.max() <= args.tol
        ok &= passed
        per_pos = np.abs(diff).max(-1)
        profile = " ".join(
            f"[{i}]={per_pos[i]:.1e}" for i in list(range(3)) + [n // 2, n - 1]
        )
        print(
            f"{name}: max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} "
            f"argmax {argmax_agree}/{n} top10 overlap {overlap:.3f} "
            f"-> {'PASS' if passed else 'FAIL'}\n"
            f"  per-position max|d|: {profile}"
        )

    print("VERDICT:", "PASS" if ok else "FAIL", f"(tol {args.tol:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
