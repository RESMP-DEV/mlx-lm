#!/usr/bin/env python3
"""Dump logits from the MLX qwen4_exp port on a tiny reference checkpoint.

Runs in the mlx env (/Users/kearm/mlx/.venv). Reads <name>/ checkpoint and
<name>-prompts.json; writes <name>-mlx.npz.
"""

import argparse
import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.qwen4_exp import Model, ModelArgs

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tiny-qwen4exp")
    ap.add_argument("--cpu", action="store_true", help="run MLX on CPU")
    args = ap.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    ckpt = HERE / args.name

    config = json.loads((ckpt / "config.json").read_text())
    args_m = ModelArgs.from_dict(config)
    model = Model(args_m)

    weights = {}
    for shard in sorted(glob.glob(str(ckpt / "*.safetensors"))):
        weights.update(mx.load(shard))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    print(f"loaded {len(weights)} tensors")

    prompts = json.loads((HERE / f"{args.name}-prompts.json").read_text())
    logits = {}
    for name, ids in prompts.items():
        out = model(mx.array([ids]))
        logits[name] = np.array(out[0], copy=False)
        print(f"{name}: logits {logits[name].shape} |abs mean| "
              f"{np.abs(logits[name]).mean():.4f}")

    np.savez(HERE / f"{args.name}-mlx.npz", **logits)
    print("wrote mlx logits")


if __name__ == "__main__":
    main()
