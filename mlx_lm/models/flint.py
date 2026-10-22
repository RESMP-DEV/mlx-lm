# Copyright © 2026 RESMP-DEV

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "flint"
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    num_hidden_layers: int = 49
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 150144
    num_experts: int = 256
    num_experts_per_tok: int = 6
    sliding_window: int = 513
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    layer_types: list[str] | None = None
    no_rope_layer_indices: list[int] | None = None
    mlp_only_layers: list[int] | None = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.skip_rope = layer_idx in set(args.no_rope_layer_indices or [])

        dim = args.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, args.rms_norm_eps)
        self.softmax_offset = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.rope = (
            None
            if self.skip_rope
            else nn.RoPE(
                self.head_dim,
                traditional=False,
                base=args.rope_theta,
            )
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: Any | None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            if self.rope is not None:
                q = self.rope(q, offset=offset)
                k = self.rope(k, offset=offset)
            k, v = cache.update_and_fetch(k, v)
        elif self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.softmax_offset,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            activation=swiglu,
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        k = self.top_k
        indices = mx.stop_gradient(mx.argpartition(scores, -k, axis=-1)[..., -k:])
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)

        y = self.switch_mlp(x, indices)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        return y + self.shared_expert(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            MLP(args.hidden_size, args.intermediate_size)
            if layer_idx in set(args.mlp_only_layers or [])
            else SparseMoeBlock(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: Any | None,
    ) -> mx.array:
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return r + self.mlp(self.post_attention_layernorm(r))


class FlintModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
    ) -> mx.array:
        x = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        full_idx = self.args.layer_types.index("full_attention")
        swa_idx = self.args.layer_types.index("sliding_attention")
        full_mask = create_attention_mask(x, cache[full_idx])
        swa_mask = create_attention_mask(
            x,
            cache[swa_idx],
            window_size=self.args.sliding_window,
        )

        for layer, layer_cache, layer_type in zip(
            self.layers, cache, self.args.layer_types
        ):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            x = layer(x, mask, layer_cache)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        if any(".mlp.switch_mlp." in key for key in weights):
            return weights

        dense_layers = set(self.args.mlp_only_layers or [])
        for layer_idx in range(self.args.num_hidden_layers):
            if layer_idx in dense_layers:
                continue
            prefix = f"model.layers.{layer_idx}.mlp"
            for name in ("gate_proj", "up_proj", "down_proj"):
                experts = [
                    weights.pop(f"{prefix}.experts.{expert}.{name}.weight")
                    for expert in range(self.args.num_experts)
                ]
                weights[f"{prefix}.switch_mlp.{name}.weight"] = mx.stack(experts)
        return weights

    def fuse_gate_up(self):
        for layer in self.model.layers:
            if isinstance(layer.mlp, SparseMoeBlock):
                layer.mlp.switch_mlp.fuse_gate_up()

    def load_weights(self, file_or_weights, strict: bool = True):
        super().load_weights(file_or_weights, strict=strict)
        self.fuse_gate_up()
        return self

    def make_cache(self):
        caches = []
        for layer_type in self.args.layer_types:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
