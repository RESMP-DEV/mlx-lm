# Copyright © 2026 Apple Inc.

"""Batched decode smoke for qwen4_exp with external PLE."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model, load_tokenizer


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def ple_table(model):
    ple = getattr(model.layers[1], "ple", None)
    return getattr(
        getattr(getattr(ple, "ple_embedding", None), "ngram_embedding", None),
        "external",
        None,
    )


def run_batch(
    model,
    prompt: mx.array,
    max_tokens: int,
    trace_rows: list | None = None,
    return_tokens: bool = False,
) -> dict:
    cache = make_prompt_cache(model)
    logits = model(prompt[:, :-1], cache=cache)
    next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(logits, next_tokens)
    logits = model(next_tokens[:, None], cache=cache)
    next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(logits, next_tokens)
    model.reset_prompt_cache(cache)
    model.clear_ple_cache()

    table = ple_table(model)
    before = table.stats() if table is not None else None
    original_gather = None
    if table is not None and trace_rows is not None:
        original_gather = table.gather

        def traced_gather(rows):
            trace_rows.append(np.asarray(rows, dtype=np.int64).copy())
            return original_gather(rows)

        table.gather = traced_gather

    latencies: list[float] = []
    generated_tokens: list[list[int]] = []
    started = time.perf_counter()
    logits = model(prompt, cache=cache)
    next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(logits, next_tokens)
    latencies.append((time.perf_counter() - started) * 1000)
    generated_tokens.append(next_tokens.tolist())

    for _ in range(max_tokens - 1):
        step_started = time.perf_counter()
        logits = model(next_tokens[:, None], cache=cache)
        next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(logits, next_tokens)
        latencies.append((time.perf_counter() - step_started) * 1000)
        generated_tokens.append(next_tokens.tolist())

    if original_gather is not None:
        table.gather = original_gather

    elapsed = sum(latencies) / 1000
    decode = latencies[1:]
    result = {
        "batch": prompt.shape[0],
        "first_token_ms": round(latencies[0], 3),
        "decode_p50_ms": round(percentile(decode, 0.50), 3),
        "decode_p95_ms": round(percentile(decode, 0.95), 3),
        "wall_seconds": round(elapsed, 3),
        "batch_tokens_per_second": round(prompt.shape[0] * max_tokens / elapsed, 3),
        "per_sequence_tokens_per_second": round(max_tokens / elapsed, 3),
    }
    if table is not None:
        after = table.stats()
        result["ple"] = {
            key: after[key] - before[key]
            for key in after
            if key
            not in {
                "rows",
                "rows_per_shard",
                "values_per_row",
                "n_shards",
                "cache_capacity",
                "cache_size",
            }
        }
    if return_tokens:
        result["generated_tokens"] = generated_tokens
    mx.clear_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--prompt", default="The quick brown fox jumps over the lazy dog."
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-batch-chunk", type=int, default=0)
    parser.add_argument("--ple-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, _ = load_model(args.model, lazy=True)
    tokenizer = load_tokenizer(args.model)
    mx.eval(model.parameters())
    base = tokenizer.encode(args.prompt)
    results = []
    for batch in args.batches:
        rows = []
        for offset in range(batch):
            ids = base.copy()
            ids[0] = (ids[0] + offset * 137) % model.args.text.vocab_size
            rows.append(ids)
        chunk_size = min(args.max_batch_chunk, batch) if args.max_batch_chunk else batch
        chunk_results = []
        chunk_tokens = []
        chunk_started = time.perf_counter()
        for chunk_index, row_start in enumerate(range(0, batch, chunk_size)):
            chunk_rows = rows[row_start : row_start + chunk_size]
            trace_rows: list[np.ndarray] = []
            chunk_result = run_batch(
                model,
                mx.array(chunk_rows),
                args.max_tokens,
                trace_rows=trace_rows,
                return_tokens=True,
            )
            chunk_result["chunk"] = chunk_index
            chunk_result["chunk_batch"] = len(chunk_rows)
            chunk_results.append(chunk_result)
            chunk_tokens.append(chunk_result.pop("generated_tokens"))
            if args.ple_trace is not None:
                suffix = (
                    f".b{batch}c{chunk_index}" if chunk_size < batch else f".b{batch}"
                )
                trace_path = args.ple_trace.with_suffix(f"{suffix}.npz")
                np.savez_compressed(
                    trace_path,
                    rows=(
                        np.concatenate([rows.reshape(-1) for rows in trace_rows])
                        if trace_rows
                        else np.empty(0, dtype=np.int64)
                    ),
                    ticket_rows=np.asarray(
                        [rows.size for rows in trace_rows], dtype=np.int64
                    ),
                    batch=np.asarray(len(chunk_rows), dtype=np.int64),
                )
                chunk_result["ple_trace"] = str(trace_path)

        if chunk_size == batch:
            result = chunk_results[0]
            result["generated_tokens"] = chunk_tokens[0]
        else:
            elapsed = time.perf_counter() - chunk_started
            steady_elapsed = sum(result["wall_seconds"] for result in chunk_results)
            result = {
                "batch": batch,
                "max_batch_chunk": chunk_size,
                "wall_seconds": round(steady_elapsed, 3),
                "setup_wall_seconds": round(elapsed - steady_elapsed, 3),
                "batch_tokens_per_second": round(
                    batch * args.max_tokens / steady_elapsed, 3
                ),
                "per_sequence_tokens_per_second": round(
                    args.max_tokens / steady_elapsed, 3
                ),
                "chunks": chunk_results,
                "generated_tokens": chunk_tokens,
            }
        results.append(result)
        print(json.dumps(result), flush=True)

    receipt = {
        "model": str(args.model),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "results": results,
        "peak_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin"
            else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "metal_peak_memory_bytes": mx.get_peak_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    model.close()


if __name__ == "__main__":
    main()
