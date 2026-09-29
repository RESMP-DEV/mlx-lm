#!/usr/bin/env python3
"""Generate a tiny random qwen4_exp reference checkpoint and logits.

Runs in the reference env (.venv-ref: torch + transformers 5.17.0, CPU).
Output: <name>/ checkpoint, <name>-prompts.json, <name>-ref.npz.

Variants: pass --set key=value (Python literal) to override text-config
fields, e.g. --set ple_layer_ids='[]'.
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from transformers import (
    Qwen4ExpConfig,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
)
from transformers.models.qwen4_exp.configuration_qwen4_exp import RopeParameters

HERE = Path(__file__).parent
VOCAB = 10_000
EOS = 1

TEXT_KWARGS = dict(
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=VOCAB,
    rms_norm_eps=1e-6,
    # Official config spelling; transformers defaults emit
    # "qwen_sparse_attention" for the same layer kind.
    layer_types=[
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    hc_count=4,
    hc_lowrank=16,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=101,
    split_ngram_parts=4,
    ple_embed_dim=64,
    ple_layer_ids=[2],
    eos_token_id=EOS,
    rope_parameters=RopeParameters(
        rope_theta=10_000_000, partial_rotary_factor=0.25
    ),
)

# Deterministic prompts, ids well inside the vocab, tokens 2..99 arbitrary.
PROMPTS = {
    "dense": [5, 9, 2, 7, 3, 8, 4, 6, 0, 11],
    "sparse": [12, 44, 7, 91, 3, 58, 20, 66, 15, 82, 29, 5, 71, 38, 94, 10, 47, 63, 26, 88, 52, 17, 76, 33],
    "eos_reset": [6, 30, 72, 19, EOS, 41, 8, 55, 27, EOS, 63, 14, 86, 37, 2, 70, 23, 49],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tiny-qwen4exp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="text-config override key=python-literal",
    )
    args = ap.parse_args()

    kwargs = dict(TEXT_KWARGS)
    for ov in args.set:
        key, _, val = ov.partition("=")
        kwargs[key] = ast.literal_eval(val)

    text = Qwen4ExpTextConfig(**kwargs)
    vision = Qwen4ExpVisionConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_channels=3,
        image_size=16,
        patch_size=8,
    )
    cfg = Qwen4ExpConfig(text_config=text, vision_config=vision)

    torch.manual_seed(args.seed)
    model = Qwen4ExpForConditionalGeneration(cfg).eval()

    rp = dict(model.config.text_config.rope_parameters)
    assert float(rp["rope_theta"]) == 1e7 and float(rp["partial_rotary_factor"]) == 0.25, (
        f"rope parameters did not stick: {rp}"
    )

    out_dir = HERE / args.name
    out_dir.mkdir(exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    print(f"saved checkpoint to {out_dir}")

    (HERE / f"{args.name}-prompts.json").write_text(json.dumps(PROMPTS, indent=2))

    logits = {}
    with torch.no_grad():
        for name, ids in PROMPTS.items():
            out = model(input_ids=torch.tensor([ids]), use_cache=False)
            logits[name] = out.logits[0].float().numpy()
            print(f"{name}: logits {logits[name].shape} |abs mean| "
                  f"{np.abs(logits[name]).mean():.4f}")

    np.savez(HERE / f"{args.name}-ref.npz", **logits)
    print("wrote reference logits")


if __name__ == "__main__":
    main()
