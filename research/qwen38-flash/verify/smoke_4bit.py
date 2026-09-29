#!/usr/bin/env python3
"""Smoke-test the real Qwen3.8-Flash-Next-4bit checkpoint through the port.

Runs in the mlx env (/Users/kearm/mlx/.venv, repo on sys.path). Replicates
mlx_lm.utils.load_model's build+quantize path but streams one shard at a
time: the stock loader uploads all ~104 GiB before its final eval, which
jetsam-kills the process when swap headroom is thin. Quantization decisions
need only name/shape/dtype metadata, so they come from safetensors headers.

Requires the Metal wired limit raised first:
sudo sysctl iogpu.wired_limit_mb=118000
"""

import argparse
import glob
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/kearm/mlx-lm")

import mlx.core as mx
import mlx.nn as nn

_DT = {
    "F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32,
    "U8": mx.uint8, "U32": mx.uint32, "I32": mx.int32, "I64": mx.int64,
}


def header_meta(shard: Path) -> dict:
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {
        name: mx.zeros((1, info["shape"][-1]), dtype=_DT[info["dtype"]])
        for name, info in hdr.items()
        if name != "__metadata__"
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default=glob.glob(
            "/Users/kearm/.cache/alphaheng/huggingface/hub/"
            "models--mlx-community--Qwen3.8-Flash-Next-4bit/snapshots/*"
        )[0],
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    path = Path(args.path)

    from mlx_lm.generate import generate
    from mlx_lm.utils import (
        _get_classes, infer_quant_config, load_config, load_tokenizer,
    )
    from mlx.utils import tree_flatten

    t0 = time.perf_counter()
    config = load_config(path)
    model_class, args_class = _get_classes(config=config)
    model = model_class(args_class.from_dict(config))

    def strip_prefix(k: str) -> str:
        if k.startswith("model.language_model."):
            return "model." + k[len("model.language_model."):]
        if k.startswith("language_model."):
            return k[len("language_model."):]
        return k

    meta: dict = {}
    shards = sorted(glob.glob(str(path / "*.safetensors")))
    for shard in shards:
        meta.update(header_meta(Path(shard)))
    # The predicate works in model-tree namespace; normalize checkpoint keys
    # the same way as the quantization overrides or nothing quantizes.
    meta = {strip_prefix(k): v for k, v in meta.items()}

    q = config.get("quantization", {})
    default = {k: v for k, v in q.items() if not isinstance(v, dict)}

    overrides = {
        strip_prefix(k): v for k, v in q.items() if isinstance(v, dict)
    }

    def class_predicate(p, m):
        if p in overrides:
            return dict(overrides[p])
        if not hasattr(m, "to_quantized"):
            return False
        if f"{p}.scales" not in meta:
            return False
        return infer_quant_config(p, m, meta)

    nn.quantize(
        model,
        group_size=default["group_size"],
        bits=default["bits"],
        mode=default.get("mode", "affine"),
        class_predicate=class_predicate,
    )
    model.eval()

    tokenizer = load_tokenizer(path)
    loaded = set()
    for i, shard in enumerate(shards):
        weights = model.sanitize(mx.load(shard))
        model.load_weights(list(weights.items()), strict=False)
        loaded.update(weights)
        for _, v in tree_flatten(model.parameters()):
            mx.eval(v)
        print(f"  shard {i + 1}/{len(shards)} resident "
              f"{mx.metal.get_active_memory() / 2**30:.1f} GiB", flush=True)
        time.sleep(1.0)

    tree = dict(tree_flatten(model.parameters()))
    missing = set(tree) - loaded
    assert not missing, f"never loaded: {sorted(missing)[:10]}"
    print(f"load: {time.perf_counter() - t0:.1f}s ({len(tree)} tensors, "
          f"{mx.metal.get_active_memory() / 2**30:.1f} GiB resident)")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="mlx").pop("input_ids")
    print(f"prompt ({inputs.shape[1]} tok): {prompt!r}")

    t0 = time.perf_counter()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                   verbose=False)
    dt = time.perf_counter() - t0
    n = args.max_tokens
    print(f"generate: {dt:.1f}s ({n / dt:.1f} tok/s incl. prefill)")
    print("output:", out)

    if mx.metal.is_available():
        print(f"peak: {mx.metal.get_peak_memory() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
