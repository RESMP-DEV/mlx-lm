# Copyright © 2026 Apple Inc.

"""Selected-row BF16 PLE reads for qwen4_exp.

The real PLE table is 320,001,536 rows of 160 BF16 values, split across 128
safetensors shards. Rows are read directly from disk, so the 95.37 GiB table
never becomes resident.
"""

from __future__ import annotations

import json
import mmap
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from .safetensors_layout import TensorLayout, read_safetensors_header


class _PleTicket:
    def __init__(
        self,
        shape,
        unique_rows,
        inverse,
        payloads,
        futures,
        cache_hits,
        disk_rows,
        started,
    ):
        self.shape = shape
        self.unique_rows = unique_rows
        self.inverse = inverse
        self.payloads = payloads
        self.futures = futures
        self.cache_hits = cache_hits
        self.disk_rows = disk_rows
        self.started = started
        self.requested = int(inverse.size)
        self.issue_ns = 0
        self.collected = False


class ExternalPleTable:
    """Read selected rows from a packed MLX checkpoint's BF16 PLE shards."""

    def __init__(
        self,
        root: str | os.PathLike,
        prefix: str | None = None,
        n_shards: int = 128,
        rows_per_shard: int | None = None,
        max_inflight: int = 32,
        cache_rows: int = 0,
        prefetch: bool = False,
        read_backend: str = "pread",
    ):
        self.root = Path(root)
        self.n_shards = int(n_shards)
        with open(self.root / "model.safetensors.index.json") as fid:
            self.weight_map = json.load(fid)["weight_map"]

        prefix_match = prefix or ".ngram_embedding.shards."
        candidates = [
            key for key in self.weight_map if prefix_match in key and ".shards." in key
        ]
        self.shard_file: list[str] = [""] * n_shards
        for key in candidates:
            index = int(key.rsplit(".shards.", 1)[1].split(".weight", 1)[0])
            if index < n_shards:
                self.shard_file[index] = self.weight_map[key]
        if any(not name for name in self.shard_file):
            missing = [i for i, name in enumerate(self.shard_file) if not name]
            raise ValueError(f"missing PLE shards {missing[:8]} under {self.root}")

        self.paths = [self.root / name for name in self.shard_file]
        self.fd: dict[int, int] = {}
        self.data_start: list[int] = [0] * n_shards
        self.tensor_rows: list[int] = [0] * n_shards
        self.values_per_shard: list[int] = [0] * n_shards
        keys_by_file: dict[str, dict[int, str]] = {}
        for key, filename in self.weight_map.items():
            if ".ngram_embedding.shards." not in key:
                continue
            shard = int(key.rsplit(".shards.", 1)[1].split(".weight", 1)[0])
            if shard < self.n_shards:
                keys_by_file.setdefault(filename, {})[shard] = key
        for filename, shard_keys in keys_by_file.items():
            path = self.root / filename
            header, data_start = read_safetensors_header(path)
            for shard, key in shard_keys.items():
                layout = TensorLayout.from_header(path, key, header, data_start)
                self.data_start[shard] = layout.data_start
                self.tensor_rows[shard] = layout.rows
                self.values_per_shard[shard] = layout.values_per_row

        if any(values != self.values_per_shard[0] for values in self.values_per_shard):
            raise ValueError("PLE shards have inconsistent row widths")
        self.values_per_row = self.values_per_shard[0]
        self.bytes_per_row = self.values_per_row * 2
        if rows_per_shard is None:
            rows_per_shard = self.tensor_rows[0]
        if any(rows != self.tensor_rows[0] for rows in self.tensor_rows):
            raise ValueError("PLE shards have inconsistent row counts")
        self.rows_per_shard = int(rows_per_shard)
        self.rows = self.rows_per_shard * self.n_shards

        self.cache_capacity = max(0, int(cache_rows))
        self.prefetch = bool(prefetch)
        self.read_backend = read_backend
        if self.read_backend not in ("pread", "mmap"):
            raise ValueError(f"unknown PLE read backend: {read_backend}")
        self.executor = ThreadPoolExecutor(max_workers=max(1, max_inflight))
        self._maps: dict[int, mmap.mmap] = {}
        self.cache: OrderedDict[int, np.ndarray] | None = (
            OrderedDict() if self.cache_capacity else None
        )
        self._cache_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "gathers": 0,
            "requested_rows": 0,
            "cache_hits": 0,
            "disk_rows": 0,
            "read_calls": 0,
            "read_bytes": 0,
            "elapsed_ns": 0,
            "issue_ns": 0,
        }

    def _open(self, shard: int) -> int:
        fd = self.fd.get(shard)
        if fd is None:
            fd = os.open(self.paths[shard], os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except (AttributeError, OSError):
                pass
            self.fd[shard] = fd
        return fd

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for mapping in self._maps.values():
            mapping.close()
        self._maps.clear()
        for fd in self.fd.values():
            os.close(fd)
        self.fd.clear()

    def clear_cache(self) -> None:
        with self._cache_lock:
            if self.cache is not None:
                self.cache.clear()

    def set_cache_capacity(self, cache_rows: int) -> None:
        capacity = max(0, int(cache_rows))
        with self._cache_lock:
            self.cache_capacity = capacity
            if capacity:
                if self.cache is None:
                    self.cache = OrderedDict()
                while len(self.cache) > capacity:
                    self.cache.popitem(last=False)
            else:
                self.cache = None

    def _read_rows(self, shard: int, rows: np.ndarray) -> np.ndarray:
        if self.read_backend == "mmap":
            return self._read_rows_mmap(shard, rows)
        """Read rows in original order, coalescing contiguous ranges."""
        fd = self._open(shard)
        out = np.empty((rows.size, self.values_per_row), dtype=np.uint16)
        start = 0
        while start < rows.size:
            stop = start + 1
            while stop < rows.size and rows[stop] == rows[stop - 1] + 1:
                stop += 1
            first = int(rows[start])
            count = int(rows[stop - 1] - first + 1)
            offset = self.data_start[shard] + first * self.bytes_per_row
            raw = os.pread(fd, count * self.bytes_per_row, offset)
            if len(raw) != count * self.bytes_per_row:
                raise RuntimeError(
                    f"PLE short read: shard {shard}, row {first}, got {len(raw)}"
                )
            self._record_stats(disk_rows=0, read_calls=1, read_bytes=len(raw))
            values = np.frombuffer(raw, dtype=np.uint16).reshape(
                count, self.values_per_row
            )
            out[start:stop] = values
            start = stop
        return out

    def _read_rows_mmap(self, shard: int, rows: np.ndarray) -> np.ndarray:
        mapping = self._maps.get(shard)
        if mapping is None:
            fd = self._open(shard)
            mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            self._maps[shard] = mapping
        out = np.empty((rows.size, self.values_per_row), dtype=np.uint16)
        base = self.data_start[shard]
        width = self.bytes_per_row
        for position, row in enumerate(rows.tolist()):
            start = base + int(row) * width
            out[position] = np.frombuffer(
                mapping[start : start + width], dtype=np.uint16
            )
        self._record_stats(
            disk_rows=0, read_calls=rows.size, read_bytes=rows.size * width
        )
        return out

    def _record_stats(self, **values: int) -> None:
        with self._stats_lock:
            for key, value in values.items():
                self._stats[key] += value

    def _cache_get(self, row: int) -> np.ndarray | None:
        if self.cache is None:
            return None
        with self._cache_lock:
            value = self.cache.get(row)
            if value is not None:
                self.cache.move_to_end(row)
            return value

    def _cache_put(self, row: int, value: np.ndarray) -> None:
        if self.cache is None:
            return
        with self._cache_lock:
            self.cache[row] = value
            self.cache.move_to_end(row)
            while len(self.cache) > self.cache_capacity:
                self.cache.popitem(last=False)

    def _read_group(self, shard: int, positions: np.ndarray, rows: np.ndarray):
        return positions, self._read_rows(shard, rows)

    def issue(self, global_rows) -> _PleTicket:
        """Validate rows and start all missing shard reads."""
        started = time.monotonic_ns()
        if not isinstance(global_rows, np.ndarray):
            global_rows = np.asarray(global_rows, dtype=np.int64)
        else:
            global_rows = global_rows.astype(np.int64)
        shape = global_rows.shape
        flat = global_rows.reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise IndexError("PLE global row out of range")

        unique_rows, inverse = np.unique(flat, return_inverse=True)
        shard_indices = unique_rows // self.rows_per_shard
        row_indices = unique_rows % self.rows_per_shard
        payloads: list[np.ndarray | None] = [None] * unique_rows.size
        positions_by_shard: dict[int, list[int]] = {}
        rows_by_shard: dict[int, list[int]] = {}
        cached_flags: list[bool] = []

        for position, shard in enumerate(shard_indices.tolist()):
            global_row = int(unique_rows[position])
            cached = self._cache_get(global_row)
            cached_flags.append(cached is not None)
            if cached is not None:
                payloads[position] = cached
                continue
            positions_by_shard.setdefault(shard, []).append(position)
            rows_by_shard.setdefault(shard, []).append(int(row_indices[position]))
        by_shard = {
            shard: (
                np.asarray(positions, dtype=np.int64),
                np.asarray(rows, dtype=np.int64),
            )
            for shard, positions in positions_by_shard.items()
            for rows in (rows_by_shard[shard],)
        }
        requested_counts = np.bincount(inverse, minlength=unique_rows.size)
        cache_hits = sum(
            int(requested_counts[position])
            for position, is_cached in enumerate(cached_flags)
            if is_cached
        )

        futures = [
            (positions, self.executor.submit(self._read_group, shard, positions, rows))
            for shard, (positions, rows) in by_shard.items()
        ]
        ticket = _PleTicket(
            shape,
            unique_rows,
            inverse,
            payloads,
            futures,
            cache_hits,
            sum(len(rows) for _, rows in by_shard.values()),
            started,
        )
        ticket.issue_ns = time.monotonic_ns() - started
        return ticket

    def collect(self, ticket: _PleTicket) -> mx.array:
        """Wait for a ticket and return BF16 rows in original order."""
        if ticket.collected:
            raise RuntimeError("PLE ticket was already collected")
        ticket.collected = True

        for _positions, future in ticket.futures:
            result_positions, restored = future.result()
            for position, value in zip(result_positions.tolist(), restored):
                ticket.payloads[position] = value
                self._cache_put(int(ticket.unique_rows[position]), value.copy())

        self._record_stats(
            gathers=1,
            requested_rows=ticket.requested,
            cache_hits=ticket.cache_hits,
            disk_rows=ticket.disk_rows,
            elapsed_ns=time.monotonic_ns() - ticket.started,
            issue_ns=ticket.issue_ns,
        )

        unique = np.empty(
            (ticket.unique_rows.size, self.values_per_row), dtype=np.uint16
        )
        for position, value in enumerate(ticket.payloads):
            if value is None:
                raise RuntimeError(
                    f"PLE row {ticket.unique_rows[position]} was not read"
                )
            unique[position] = value
        raw = unique[ticket.inverse]

        bits = mx.array(raw.reshape(-1))
        values = bits.view(mx.bfloat16).reshape(*ticket.shape, self.values_per_row)
        mx.eval(values)
        return values

    def gather(self, global_rows) -> mx.array:
        """Read selected logical rows. Returns an MLX array with dtype bfloat16."""
        return self.collect(self.issue(global_rows))

    def stats(self) -> dict:
        with self._stats_lock:
            stats = dict(self._stats)
        stats.update(
            {
                "rows": self.rows,
                "rows_per_shard": self.rows_per_shard,
                "values_per_row": self.values_per_row,
                "n_shards": self.n_shards,
                "cache_capacity": self.cache_capacity,
                "cache_size": len(self.cache) if self.cache is not None else 0,
            }
        )
        return stats
