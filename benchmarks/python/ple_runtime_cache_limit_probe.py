# Copyright © 2026 Apple Inc.

"""Run ple_runtime_smoke with a Metal cache limit set after model load."""

from __future__ import annotations

import os
import sys

import mlx.core as mx

import ple_runtime_smoke as smoke


def main() -> None:
    limit = int(os.environ["PLE_CACHE_LIMIT_BYTES"])
    original_load = smoke.load_model

    def load_model(path, lazy=True):
        model, config = original_load(path, lazy=lazy)
        previous = mx.set_cache_limit(limit)
        print(
            {
                "cache_limit_bytes": limit,
                "previous_cache_limit_bytes": previous,
                "cache_memory_after_load": mx.get_cache_memory(),
            },
            file=sys.stderr,
            flush=True,
        )
        return model, config

    smoke.load_model = load_model
    smoke.main()


if __name__ == "__main__":
    main()
