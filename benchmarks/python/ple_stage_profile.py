# Copyright © 2026 Apple Inc.

"""Synchronized stage profile for qwen4_exp with external PLE."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from mlx_lm.models import qwen4_exp as model_module
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen4_exp import (
    Attention,
    GatedDeltaNet,
    GatedResidual,
    PLELayer,
    QSAIndexer,
    SparseMoeBlock,
)
from mlx_lm.utils import load_model, load_tokenizer


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median_ms": round(percentile(values, 0.50) / 1e6, 3),
        "mean_ms": round(sum(values) / len(values) / 1e6, 3),
        "p95_ms": round(percentile(values, 0.95) / 1e6, 3),
        "max_ms": round(max(values) / 1e6, 3),
        "total_ms_per_token": round(sum(values) / 1e6, 3),
    }


def evaluate(result):
    if isinstance(result, tuple):
        for item in result:
            evaluate(item)
    elif isinstance(result, mx.array):
        mx.eval(result)


def install_profiler() -> dict[list[float]]:
    samples = defaultdict(list)

    def wrap_class(cls, stage):
        original = cls.__call__

        def timed(self, *args, **kwargs):
            started = time.perf_counter_ns()
            result = original(self, *args, **kwargs)
            evaluate(result)
            samples[stage].append(time.perf_counter_ns() - started)
            return result

        cls.__call__ = timed
        return original

    originals = [
        wrap_class(QSAIndexer, "qsa"),
        wrap_class(Attention, "attention"),
        wrap_class(GatedDeltaNet, "gdn"),
        wrap_class(SparseMoeBlock, "moe"),
        wrap_class(GatedResidual, "residual"),
        wrap_class(PLELayer, "ple"),
    ]

    original_inject = model_module._compiled_inject_block

    def timed_inject(*args, **kwargs):
        started = time.perf_counter_ns()
        result = original_inject(*args, **kwargs)
        evaluate(result)
        samples["injection"].append(time.perf_counter_ns() - started)
        return result

    model_module._compiled_inject_block = timed_inject

    def restore():
        for cls, original in zip(
            (
                QSAIndexer,
                Attention,
                GatedDeltaNet,
                SparseMoeBlock,
                GatedResidual,
                PLELayer,
            ),
            originals,
        ):
            cls.__call__ = original
        model_module._compiled_inject_block = original_inject

    return samples, restore


def warm_prompt(model, prompt: mx.array) -> None:
    cache = make_prompt_cache(model)
    logits = model(prompt[:-1][None], cache=cache)
    mx.eval(logits)
    logits = model(prompt[-1:][None], cache=cache)
    mx.eval(logits)
    model.reset_prompt_cache(cache)
    model.clear_ple_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, config = load_model(args.model, lazy=True)
    model.compile_decode()
    mx.eval(model.parameters())
    tokens = load_tokenizer(args.model)
    prompt = mx.array(tokens.encode(args.prompt_file.read_text()))
    mx.eval(prompt)
    warm_prompt(model, prompt)

    samples, restore = install_profiler()
    token_latency = []
    generated = []
    cache = make_prompt_cache(model)
    logits = model(prompt[:-1][None], cache=cache)
    mx.eval(logits)
    samples.clear()
    try:
        for index in range(args.tokens):
            started = time.perf_counter_ns()
            input_token = prompt[-1:] if index == 0 else mx.array([generated[-1]])
            logits = model(input_token[None], cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(logits, token)
            token_latency.append(time.perf_counter_ns() - started)
            generated.append(token.item())
    finally:
        restore()

    receipt = {
        "model": str(args.model),
        "prompt_file": str(args.prompt_file),
        "prompt_tokens": int(prompt.shape[0]),
        "profiled_tokens": len(generated),
        "method": "Synchronized monkeypatch timing; each stage result is forced with mx.eval.",
        "stages": {name: summarize(values) for name, values in sorted(samples.items())},
        "token_latency_ms": summarize(token_latency),
        "prefill_step_size": 512,
        "ple_backend": config.get("text_config", {}).get("ple_backend"),
        "non_claims": [
            "Serialized stage totals are not steady-state decode latency.",
            "Relative shares, not absolute command latencies, guide optimization.",
            "No Optane result is claimed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    model.close()


if __name__ == "__main__":
    main()
