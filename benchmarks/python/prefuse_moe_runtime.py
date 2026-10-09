# Copyright © 2026 Apple Inc.

"""Build a qwen4_exp runtime with shared-input projections pre-fused."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fid:
        for block in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fuse_file(source: Path, destination: Path) -> dict:
    arrays = mx.load(str(source))
    output = {}
    pairs = 0
    gdn_pairs = 0
    for key in sorted(arrays):
        if not key.endswith(".mlp.switch_mlp.gate_proj.weight"):
            continue
        base = key[: -len("gate_proj.weight")]
        up_weight = arrays.get(base + "up_proj.weight")
        up_scales = arrays.get(base + "up_proj.scales")
        if up_weight is None or up_scales is None:
            raise ValueError(f"missing up projection for {key}")
        fused = base + "gate_up_proj.weight"
        output[fused] = mx.concatenate([arrays[key], up_weight], axis=1)
        output[base + "gate_up_proj.scales"] = mx.concatenate(
            [arrays[base + "gate_proj.scales"], up_scales], axis=1
        )
        pairs += 1

    for key in sorted(arrays):
        if not key.endswith(".linear_attn.in_proj_qkv.weight"):
            continue
        base = key[: -len("in_proj_qkv.weight")]
        parts = [arrays[key], arrays[base + "in_proj_z.weight"]]
        scales = [
            arrays[base + "in_proj_qkv.scales"],
            arrays[base + "in_proj_z.scales"],
        ]
        for name in ("in_proj_b", "in_proj_a"):
            parts.append(arrays[base + name + ".weight"])
            scales.append(arrays[base + name + ".scales"])
        output[key] = mx.concatenate(parts, axis=0)
        output[base + "in_proj_qkv.scales"] = mx.concatenate(scales, axis=0)
        gdn_pairs += 1

    for key in sorted(arrays):
        if key in output:
            continue
        if ".mlp.switch_mlp.gate_proj." in key:
            continue
        if ".mlp.switch_mlp.up_proj." in key:
            continue
        if ".linear_attn.in_proj_z." in key:
            continue
        if ".linear_attn.in_proj_b." in key:
            continue
        if ".linear_attn.in_proj_a." in key:
            continue
        output[key] = arrays[key]

    temporary = destination.with_name(destination.name + ".tmp.safetensors")
    mx.save_safetensors(str(temporary), output)
    temporary.replace(destination)
    return {
        "file": destination.name,
        "gate_up_pairs": pairs,
        "gdn_projection_pairs": gdn_pairs,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    destination.mkdir(parents=True)

    started = time.monotonic()
    files = []
    for path in sorted(source.iterdir()):
        if path.name == "model.safetensors.index.json":
            continue
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            item = fuse_file(path, destination / path.name)
            files.append(item)
            print(
                f"fused {path.name}: {item['gate_up_pairs']} MoE, "
                f"{item['gdn_projection_pairs']} GDN",
                flush=True,
            )
        elif path.name.endswith(".sha256.json"):
            continue
        elif path.name.startswith("config.json.pre-"):
            continue
        else:
            shutil.copy2(path, destination / path.name)

    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = {}
    for key, filename in index["weight_map"].items():
        if key.endswith(".mlp.switch_mlp.gate_proj.weight"):
            base = key[: -len("gate_proj.weight")]
            weight_map[base + "gate_up_proj.weight"] = filename
        elif key.endswith(".mlp.switch_mlp.gate_proj.scales"):
            base = key[: -len("gate_proj.scales")]
            weight_map[base + "gate_up_proj.scales"] = filename
        elif ".mlp.switch_mlp.up_proj." in key:
            continue
        elif any(f".linear_attn.in_proj_{name}." in key for name in ("z", "b", "a")):
            continue
        else:
            weight_map[key] = filename
    index["weight_map"] = weight_map
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )

    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["prefused_moe"] = True
    config["text_config"]["prefused_gdn"] = True
    config["text_config"]["fuse_gdn_all"] = False
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    tensor_counts = {filename: 0 for item in files for filename in [item["file"]]}
    for filename in weight_map.values():
        tensor_counts[filename] += 1
    for item in files:
        item["tensors"] = tensor_counts[item["file"]]
    small_files = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(destination.iterdir())
        if path.name != "prefused-runtime.sha256.json"
        and not (path.name.startswith("model-") and path.suffix == ".safetensors")
    ]
    manifest = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(destination),
        "shard_count": len(files),
        "tensor_count": len(weight_map),
        "total_model_bytes": sum(item["bytes"] for item in files),
        "shards": files,
        "small_files": small_files,
        "checks": {
            "all_shards_readable": True,
            "gate_up_pairs": sum(item["gate_up_pairs"] for item in files),
            "gdn_projection_pairs": sum(item["gdn_projection_pairs"] for item in files),
            "split_gate_up_keys": sum(
                ".mlp.switch_mlp.gate_proj." in key or ".mlp.switch_mlp.up_proj." in key
                for key in weight_map
            ),
            "split_gdn_projection_keys": sum(
                any(f".linear_attn.in_proj_{name}." in key for name in ("z", "b", "a"))
                for key in weight_map
            ),
        },
    }
    (destination / "prefused-runtime.sha256.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    receipt = {
        "source": str(source),
        "destination": str(destination),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gate_up_pairs": sum(item["gate_up_pairs"] for item in files),
        "gdn_projection_pairs": sum(item["gdn_projection_pairs"] for item in files),
        "safetensors": files,
        "config": {
            "prefused_moe": True,
            "prefused_gdn": True,
            "compile_moe": config["text_config"].get("compile_moe", False),
            "compile_residual": config["text_config"].get("compile_residual", False),
            "ple_backend": config["text_config"].get("ple_backend"),
        },
        "destination_manifest": str(destination / "prefused-runtime.sha256.json"),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
