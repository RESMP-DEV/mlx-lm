# Copyright © 2026 RESMP-DEV

import mlx.core as mx

from mlx_lm.models import flint
from mlx_lm.models.cache import KVCache, RotatingKVCache


def tiny_args():
    return flint.ModelArgs(
        hidden_size=32,
        intermediate_size=16,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        num_experts=3,
        num_experts_per_tok=2,
        sliding_window=5,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        no_rope_layer_indices=[0, 3],
        mlp_only_layers=[0],
    )


def test_flint_forward_and_hybrid_cache():
    model = flint.Model(tiny_args())
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    logits = model(inputs)
    mx.eval(logits)
    assert logits.shape == (1, 3, 64)

    assert model.model.layers[0].self_attn.rope is None
    assert model.model.layers[1].self_attn.rope is not None
    caches = model.make_cache()
    assert isinstance(caches[0], KVCache)
    assert isinstance(caches[1], RotatingKVCache)
    assert caches[1].max_size == 5

    first = model(inputs[:, :2], cache=caches)
    mx.eval(first)
    second = model(inputs[:, 2:3], cache=caches)
    mx.eval(second)
    assert all(cache.offset == 3 for cache in caches)


def test_flint_sanitize_stacks_routed_experts():
    args = tiny_args()
    model = flint.Model(args)
    weights = {
        "model.layers.1.mlp.gate.weight": mx.zeros((args.num_experts, args.hidden_size))
    }
    for layer in (1, 2, 3):
        for expert in range(args.num_experts):
            for name, out_dim in (
                ("gate_proj", args.moe_intermediate_size),
                ("up_proj", args.moe_intermediate_size),
                ("down_proj", args.hidden_size),
            ):
                weights[f"model.layers.{layer}.mlp.experts.{expert}.{name}.weight"] = (
                    mx.zeros((out_dim, args.hidden_size))
                )

    sanitized = model.sanitize(weights)
    key = "model.layers.1.mlp.switch_mlp.gate_proj.weight"
    assert sanitized[key].shape == (
        args.num_experts,
        args.moe_intermediate_size,
        args.hidden_size,
    )
    assert not any(".experts." in key for key in sanitized)


def test_flint_sanitize_accepts_already_fused_experts():
    args = tiny_args()
    model = flint.Model(args)
    weights = {
        "model.layers.1.mlp.switch_mlp.gate_proj.weight": mx.zeros(
            (args.num_experts, args.moe_intermediate_size, args.hidden_size)
        )
    }
    assert model.sanitize(weights) is weights
