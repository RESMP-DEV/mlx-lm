# Copyright © 2026 Apple Inc.

"""Build a qwen4_exp runtime with external BF16 PLE storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

from ..models.safetensors_layout import TensorLayout, read_safetensors_header


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _is_ple_key(key: str) -> bool:
    return ".ple.ple_embedding.ngram_embedding.shards." in key


def _copy_range(source, destination, start: int, size: int, chunk: int = 8 << 20):
    source.seek(start)
    remaining = size
    while remaining:
        data = source.read(min(chunk, remaining))
        if not data:
            raise RuntimeError(f"short safetensors data read: {source.name}")
        destination.write(data)
        remaining -= len(data)


def _prune_safetensors(source: Path, destination: Path, removed: set[str]) -> int:
    header, data_start = read_safetensors_header(source)
    output = {"__metadata__": header.get("__metadata__", {})}
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for key, entry in header.items():
        if key == "__metadata__" or key in removed:
            continue
        begin, end = entry["data_offsets"]
        size = end - begin
        output[key] = {**entry, "data_offsets": [cursor, cursor + size]}
        ranges.append((data_start + begin, size))
        cursor += size

    raw_header = json.dumps(output, separators=(",", ":")).encode()
    padding = (-len(raw_header)) % 8
    raw_header += b" " * padding
    header_length = len(raw_header)
    with source.open("rb") as input_file, destination.open("wb") as output_file:
        output_file.write(struct.pack("<Q", header_length))
        output_file.write(raw_header)
        for start, size in ranges:
            _copy_range(input_file, output_file, start, size)
    return 8 + header_length + cursor


def build_external_runtime(
    source: str | os.PathLike,
    destination: str | os.PathLike,
    external_root: str | os.PathLike | None = None,
    max_inflight: int = 32,
    cache_rows: int = 0,
    hash_backend: str = "auto",
) -> dict:
    source = Path(source).resolve()
    destination = Path(destination)
    external_root = Path(external_root or source).resolve()
    if destination.exists():
        raise FileExistsError(f"runtime destination already exists: {destination}")

    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("qwen4_exp config has no text_config")

    n_shards = int(text_config.get("split_ngram_parts", 0))
    if n_shards <= 0:
        raise ValueError("invalid split_ngram_parts")
    ngram_heads = (int(text_config.get("ngram_size", 0)) - 1) * int(
        text_config.get("heads_per_ngram", 0)
    )
    if ngram_heads <= 0:
        raise ValueError("invalid PLE head count")
    ple_keys = [key for key in index["weight_map"] if _is_ple_key(key)]
    if len(ple_keys) != n_shards:
        raise ValueError(f"found {len(ple_keys)} PLE shards, expected {n_shards}")
    rows_per_shard = None
    values_per_row = None
    keys_by_file: dict[str, list[str]] = {}
    for key in ple_keys:
        keys_by_file.setdefault(index["weight_map"][key], []).append(key)
    for filename, keys in keys_by_file.items():
        path = external_root / filename
        header, data_start = read_safetensors_header(path)
        for key in keys:
            layout = TensorLayout.from_header(path, key, header, data_start)
            if rows_per_shard is None:
                rows_per_shard = layout.rows
                values_per_row = layout.values_per_row
            elif (
                layout.rows != rows_per_shard or layout.values_per_row != values_per_row
            ):
                raise ValueError("PLE shards have inconsistent shapes")
    if values_per_row * ngram_heads != int(text_config.get("ple_embed_dim", -1)):
        raise ValueError("PLE row width does not match ple_embed_dim")
    kept = {
        key: value for key, value in index["weight_map"].items() if not _is_ple_key(key)
    }
    needed_files = sorted(set(kept.values()))

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.samefile(external_root):
        raise ValueError("runtime destination cannot replace the external PLE root")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        keys_by_file: dict[str, set[str]] = {}
        for key in index["weight_map"]:
            if _is_ple_key(key):
                keys_by_file.setdefault(index["weight_map"][key], set()).add(key)
        pruned_files = []
        runtime_bytes = 0
        for filename in needed_files:
            source_file = source / filename
            removed = keys_by_file.get(filename, set())
            if removed:
                runtime_bytes += _prune_safetensors(
                    source_file, temporary / filename, removed
                )
                pruned_files.append(filename)
            else:
                os.link(source_file, temporary / filename)
                runtime_bytes += source_file.stat().st_size
        for path in source.iterdir():
            if (
                not path.is_file()
                or path.suffix == ".safetensors"
                or path.name
                in {
                    "config.json",
                    "model.safetensors.index.json",
                    "model.sha256.json",
                }
            ):
                continue
            shutil.copy2(path, temporary / path.name)

        index["weight_map"] = {key: kept[key] for key in sorted(kept)}
        index.setdefault("metadata", {})["total_size"] = runtime_bytes
        text_config["ple_backend"] = {
            "root": str(external_root),
            "n_shards": n_shards,
            "rows_per_shard": rows_per_shard,
            "max_inflight": max_inflight,
            "cache_rows": cache_rows,
            "hash_backend": hash_backend,
        }
        _write_json(temporary / "config.json", config)
        _write_json(temporary / "model.safetensors.index.json", index)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "source": str(source),
        "destination": str(destination),
        "external_root": str(external_root),
        "removed_ple_keys": len(ple_keys),
        "runtime_shards": len(needed_files),
        "pruned_shards": pruned_files,
        "runtime_bytes": runtime_bytes,
        "rows_per_shard": rows_per_shard,
        "values_per_row": values_per_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--cache-rows", type=int, default=0)
    parser.add_argument(
        "--hash-backend", choices=("auto", "host", "mlx"), default="auto"
    )
    args = parser.parse_args()
    receipt = build_external_runtime(
        args.source,
        args.destination,
        args.external_root,
        args.max_inflight,
        args.cache_rows,
        args.hash_backend,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
