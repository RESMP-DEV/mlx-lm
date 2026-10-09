# Copyright © 2026 Apple Inc.

"""Fixed-prompt smoke for a qwen4_exp runtime with external BF16 PLE."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import greedy_sampler
from mlx_lm.utils import load_model, load_tokenizer


def resident_bytes(model) -> int:
    return sum(array.nbytes for _, array in tree_flatten(model.parameters()))


def ple_tables(model) -> list:
    tables = []
    for layer in model.model.layers:
        ple = getattr(layer, "ple", None)
        embedding = getattr(
            getattr(ple, "ple_embedding", None), "ngram_embedding", None
        )
        table = getattr(embedding, "external", None)
        if table is not None:
            tables.append(table)
    return tables


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--prompt", default="The quick brown fox jumps over the lazy dog."
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefill-step-size", type=int, default=0)
    parser.add_argument("--warmup-parameters", action="store_true")
    parser.add_argument("--warmup-forward", action="store_true")
    parser.add_argument("--compile-decode", action="store_true")
    parser.add_argument("--weight-before-down", action="store_true")
    parser.add_argument("--greedy-fast", action="store_true")
    parser.add_argument("--wired-limit-gb", type=float, default=0.0)
    parser.add_argument("--metal-capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wired_limit = int(args.wired_limit_gb * 1024**3)
    previous_wired_limit = mx.metal.set_wired_limit(wired_limit) if wired_limit else 0

    started = time.monotonic()
    model, config = load_model(args.model, lazy=True)
    weight_before_down = args.weight_before_down
    for layer in model.layers:
        layer.mlp.switch_mlp.weight_before_down = weight_before_down
    tables = ple_tables(model)
    load_seconds = time.monotonic() - started
    tokens = load_tokenizer(args.model)
    prompt = mx.array(tokens.encode(args.prompt))
    mx.eval(prompt)
    config_compile = any(
        config.get("text_config", {}).get(name, False)
        for name in (
            "compile_moe",
            "compile_residual",
            "compile_injection",
            "compile_attention_output",
            "compile_gdn_output",
        )
    )
    compile_decode = (args.compile_decode or config_compile) and hasattr(
        model, "compile_decode"
    )
    if compile_decode:
        model.compile_decode()

    warmup_started = time.monotonic()
    if args.warmup_parameters:
        mx.eval(model.parameters())
    parameter_warmup_seconds = time.monotonic() - warmup_started

    forward_started = time.monotonic()
    if args.warmup_forward:
        warmup_cache = make_prompt_cache(model)
        prefill = prompt[:-1] if prompt.shape[0] > 1 else prompt
        mx.eval(model(prefill[None], cache=warmup_cache))
        mx.eval(model(prompt[-1:][None], cache=warmup_cache))
        model.reset_prompt_cache(warmup_cache)
        mx.eval(model(prefill[None], cache=warmup_cache))
        mx.eval(model(prompt[-1:][None], cache=warmup_cache))
        warmup_tokens = generate_step(
            prompt,
            model,
            max_tokens=1,
            sampler=greedy_sampler,
            prompt_cache=warmup_cache,
        )
        next(warmup_tokens)
        model.reset_prompt_cache(warmup_cache)
        model.clear_ple_cache()
    forward_warmup_seconds = time.monotonic() - forward_started

    before = [table.stats() for table in tables]
    token_latencies: list[float] = []
    token_ple_latencies: list[float] = []
    generate_started = time.monotonic()
    generated_ids = []
    mx.reset_peak_memory()
    if args.metal_capture is not None:
        mx.metal.start_capture(str(args.metal_capture))
    token_generator = generate_step(
        prompt,
        model,
        max_tokens=args.max_tokens,
        sampler=greedy_sampler,
        greedy_fast=args.greedy_fast,
        prefill_step_size=(
            args.prefill_step_size if args.prefill_step_size > 0 else 512
        ),
    )
    while True:
        step_started = time.perf_counter_ns()
        step_before = [table.stats() for table in tables]
        try:
            yielded_token, _ = next(token_generator)
        except StopIteration:
            break
        token = (
            yielded_token.item()
            if hasattr(yielded_token, "item")
            else int(yielded_token)
        )
        token_latencies.append((time.perf_counter_ns() - step_started) / 1e6)
        step_after = [table.stats() for table in tables]
        token_ple_latencies.append(
            sum(
                step_after[i]["elapsed_ns"] - step_before[i]["elapsed_ns"]
                for i in range(len(tables))
            )
            / 1e6
        )
        if token in tokens.eos_token_ids:
            break
        generated_ids.append(token)
        if len(generated_ids) == args.max_tokens:
            break
    if args.metal_capture is not None:
        mx.metal.stop_capture()
    generate_seconds = time.monotonic() - generate_started
    after = [table.stats() for table in tables]
    output = mx.array([prompt.tolist() + generated_ids])
    text = tokens.decode(output[0].tolist())

    prompt_tokens = int(prompt.shape[0])
    output_tokens = int(output.shape[1])
    generated_tokens = len(generated_ids)
    decode_latencies = token_latencies[1:]
    decode_median = percentile(decode_latencies, 0.50)
    steady_decode_latencies = [
        value for value in decode_latencies if value <= decode_median * 2
    ]
    receipt = {
        "model": str(args.model),
        "metal_capture": (
            str(args.metal_capture) if args.metal_capture is not None else None
        ),
        "prompt": args.prompt,
        "output_tokens": output_tokens,
        "prefill_step_size": (
            args.prefill_step_size if args.prefill_step_size > 0 else 512
        ),
        "text": text,
        "load_seconds": round(load_seconds, 3),
        "parameter_warmup_seconds": round(parameter_warmup_seconds, 3),
        "forward_warmup_seconds": round(forward_warmup_seconds, 3),
        "generate_seconds": round(generate_seconds, 3),
        "tokens_per_second": round(
            generated_tokens / generate_seconds if generate_seconds else 0.0, 3
        ),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens_per_second": round(
            len(decode_latencies) / (sum(decode_latencies) / 1000)
            if decode_latencies
            else 0.0,
            3,
        ),
        "resident_parameter_bytes": resident_bytes(model),
        "metal_peak_memory_bytes": mx.get_peak_memory(),
        "peak_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin"
            else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "ple_layers": len(tables),
        "ple_backend": config.get("text_config", {}).get("ple_backend"),
        "prefused_moe": bool(config.get("text_config", {}).get("prefused_moe", False)),
        "prefused_gdn": bool(config.get("text_config", {}).get("prefused_gdn", False)),
        "compile_decode": compile_decode,
        "compile_residual": bool(
            config.get("text_config", {}).get("compile_residual", False)
        ),
        "compile_injection": bool(
            config.get("text_config", {}).get("compile_injection", False)
        ),
        "compile_attention_output": bool(
            config.get("text_config", {}).get("compile_attention_output", False)
        ),
        "fuse_gdn_all": bool(config.get("text_config", {}).get("fuse_gdn_all", False)),
        "compile_gdn_output": bool(
            config.get("text_config", {}).get("compile_gdn_output", False)
        ),
        "weight_before_down": weight_before_down,
        "greedy_fast": args.greedy_fast,
        "wired_limit_bytes": wired_limit,
        "previous_wired_limit_bytes": previous_wired_limit,
        "ple_stats": [
            {key: after[i][key] - before[i][key] for key in after[i]}
            for i in range(len(tables))
        ],
        "token_latency_ms": [round(value, 3) for value in token_latencies],
        "token_ple_latency_ms": [round(value, 3) for value in token_ple_latencies],
        "prompt_to_first_token_ms": round(
            token_latencies[0] if token_latencies else 0.0, 3
        ),
        "decode_latency_ms": {
            "count": len(decode_latencies),
            "p50": round(decode_median, 3),
            "p95": round(percentile(decode_latencies, 0.95), 3),
            "max": round(max(decode_latencies, default=0.0), 3),
        },
        "steady_decode_latency_ms": {
            "outlier_count": len(decode_latencies) - len(steady_decode_latencies),
            "count": len(steady_decode_latencies),
            "p50": round(percentile(steady_decode_latencies, 0.50), 3),
            "p95": round(percentile(steady_decode_latencies, 0.95), 3),
            "max": round(max(steady_decode_latencies, default=0.0), 3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    model.close()


if __name__ == "__main__":
    main()
