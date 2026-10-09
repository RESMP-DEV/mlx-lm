# Copyright © 2026 Apple Inc.

"""Parses safetensors tensor byte ranges without loading tensor payloads."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorLayout:
    path: Path
    key: str
    data_start: int
    rows: int
    values_per_row: int
    dtype: str

    @classmethod
    def from_header(
        cls,
        path: str | os.PathLike,
        key: str,
        header: dict,
        data_start: int,
    ) -> TensorLayout:
        path = Path(path)
        if key not in header:
            raise KeyError(f"missing tensor {key!r} in {path}")
        entry = header[key]
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"unexpected tensor shape for {key!r} in {path}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"missing tensor offsets for {key!r} in {path}")
        if entry.get("dtype") != "BF16":
            raise ValueError(f"{key!r} is {entry.get('dtype')!r}, expected BF16")
        size = offsets[1] - offsets[0]
        if size != shape[0] * shape[1] * 2:
            raise ValueError(f"invalid byte range for {key!r} in {path}")
        return cls(
            path,
            key,
            data_start + offsets[0],
            int(shape[0]),
            int(shape[1]),
            entry["dtype"],
        )

    @classmethod
    def parse(cls, path: str | os.PathLike, key: str) -> TensorLayout:
        header, data_start = read_safetensors_header(path)
        return cls.from_header(path, key, header, data_start)


def read_safetensors_header(path: str | os.PathLike) -> tuple[dict, int]:
    path = Path(path)
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = source.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"short safetensors header: {path}")
        return json.loads(raw_header), 8 + header_length
