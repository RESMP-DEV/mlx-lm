# Copyright © 2026 Apple Inc.

# MLX port of Qwen3.8-Flash-Next (HF model_type: qwen4_exp)
# New compared to qwen3_next: QSA sparse attention, gated residual
# (hyper-connections), sharded n-gram / PLE embedding, split deltanet projections.

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, BatchKVCache, KVCache, _BaseCache, dynamic_roll
from .gated_delta import gated_delta_update, normalize_qk
from .qwen4_exp_ple import ExternalPleTable
from .switch_layers import SwitchGLU


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: list = field(default_factory=list)
    full_attention_interval: int = 4
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # n-gram / PLE
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ple_embed_dim: int = 2560
    ple_layer_ids: list = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    seed: int = 0
    eos_token_id: Any = 248044
    ple_backend: dict | None = None
    partial_rotary_factor: float = 0.25
    rope_parameters: dict = field(default_factory=dict)
    rope_theta: float = 10_000_000.0
    tie_word_embeddings: bool = False
    compile_moe: bool = False
    compile_residual: bool = False
    compile_injection: bool = False
    compile_attention_output: bool = False
    fuse_gdn_all: bool = False
    compile_gdn_output: bool = False
    prefused_moe: bool = False
    prefused_gdn: bool = False
    gdn_packed_t1: bool = False
    gdn_packed_t1_max_context: int = 0


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    quantization: Any = None

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        rp = self.text.rope_parameters or {}
        self.text.rope_theta = float(rp.get("rope_theta", self.text.rope_theta))
        self.text.partial_rotary_factor = float(
            rp.get("partial_rotary_factor", self.text.partial_rotary_factor)
        )
        if self.text.layer_types:
            # transformers defaults emit "qwen_sparse_attention" for the layer
            # kind the official config spells "full_attention".
            self.text.layer_types = [
                "full_attention" if t == "qwen_sparse_attention" else t
                for t in self.text.layer_types
            ]
        else:
            n, k = self.text.num_hidden_layers, self.text.full_attention_interval
            self.text.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(n)
            ]


# --------------------------------------------------------------------------- norms


class RMSNorm(nn.Module):
    """Zero-centered RMSNorm: y = norm(x) * (1 + weight).

    The checkpoint stores zero-centered weights, so `Qwen4ExpTextRMSNorm` scales
    by `1 + weight` and initializes the weight to 0. Only the gated variant used
    by the delta net is conventional, see `RMSNormGated`.

    Hyper-connections normalize each of the hc_count streams separately, hence the
    reshape: one weight of size hc_count*hidden, but one statistic per stream. The
    scale applies to the flat vector, after the reshape, like the reference.
    """

    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        # ones, not zeros: the reference scales by `1 + w` and `sanitize` folds that 1
        # into the weight, so an unloaded module has to be the identity.
        self.weight = mx.ones(dim)
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"dim {dim} is not divisible by group_size {group_size}")

    def __call__(self, x: mx.array) -> mx.array:
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps).reshape(shape)
        return x * self.weight


class RMSNormGated(nn.Module):
    """Conventional RMSNorm: `Qwen4ExpTextRMSNormGated` scales by `weight` alone
    and initializes it to 1, unlike the non-gated norm above."""

    def __init__(self, dim: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return out.astype(x.dtype)
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        g = act(gate.astype(mx.float32))
        return (g * out.astype(mx.float32)).astype(x.dtype)


# ------------------------------------------------------------------- rope / helpers


def _rope_partial(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rope to the first `rotary_dim` dimensions only."""
    d = cos.shape[-1]
    # cos/sin are computed in float32: without this cast they promote x and the
    # whole attention falls back to float32.
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    xr, xp = x[..., :d], x[..., d:]
    half = d // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    xr = xr * cos + rot * sin
    return mx.concatenate([xr, xp], axis=-1) if xp.shape[-1] else xr


class RotaryEmbedding:
    def __init__(self, dim: int, base: float):
        self.dim = dim
        self.inv_freq = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)

    def __call__(self, positions: mx.array):
        # positions: (B, T) -> cos/sin (B, T, dim)
        freqs = positions.astype(mx.float32)[..., None] * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def _positions(offset: Union[int, mx.array], S: int) -> mx.array:
    """Positions of `S` tokens from `offset`, (1, S) or (B, S).

    `offset` is an int with the usual caches, and one position per slot with a
    batched cache (see `BatchKVCache.offset`).
    """
    if isinstance(offset, mx.array):
        return offset.reshape(-1, 1) + mx.arange(S)
    return mx.arange(offset, offset + S)[None]


def _left_padding(cache) -> Optional[mx.array]:
    """Left padding of a batched cache, per row, or None if the rows align."""
    pad = getattr(cache, "left_padding", None)
    if pad is None or pad.max().item() == 0:
        return None
    return pad


# ------------------------------------------------------------------------ QSA


class QSAIndexer(nn.Module):
    """Select, per query, a budget of compressed key blocks.

    The reference PyTorch implementation loops over (batch, query); here everything
    is vectorized: pooled keys do not depend on the query, so they are computed once
    and followed by a per-row top-k.
    """

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(self, x, rope, cache, offset, left_padding=None) -> Optional[mx.array]:
        B, S, _ = x.shape
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)

        if cache is not None:
            raw_k = cache.update(raw_k)
        kv_len = raw_k.shape[1]

        # No sparsification possible: every visible token fits in the budget, so the
        # top-k would keep them all. The usual causal mask is enough.
        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // self.compress_ratio
        pad = left_padding
        if pad is None:
            pooled = raw_k[:, : n_blocks * self.compress_ratio].reshape(
                B, n_blocks, self.compress_ratio, self.head_dim
            )
        else:
            # A block groups the tokens of one row, so it starts at the first
            # real token of that row, not at column 0 of the shared buffer.
            # Credit: Blaizzy/mlx-vlm#2028.
            cols = (
                pad[:, None, None]
                + (mx.arange(n_blocks) * self.compress_ratio)[None, :, None]
                + mx.arange(self.compress_ratio)[None, None, :]
            ).reshape(B, -1)
            # A block that runs past the buffer is masked out below.
            cols = mx.minimum(cols, kv_len - 1)
            pooled = mx.take_along_axis(
                raw_k,
                mx.broadcast_to(cols[..., None], (*cols.shape, self.head_dim)),
                axis=1,
            ).reshape(B, n_blocks, self.compress_ratio, self.head_dim)
        pooled = self.k_layernorm(
            pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
        )

        # Block n holds the logical positions n * compress_ratio and up in every
        # row, so the padding does not move the block rope.
        block_starts = mx.arange(n_blocks) * self.compress_ratio
        cos_k, sin_k = rope(block_starts[None, :])
        pooled = _rope_partial(pooled, cos_k, sin_k)

        q_pos = _positions(offset, S)
        cos_q, sin_q = rope(q_pos)
        q = self.q_layernorm(q)
        q = _rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])

        # scores: sum over heads of relu(q.k), per block
        scores = mx.einsum(
            "bshd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)

        # A block is only a candidate if it lies entirely in the query's past.
        # Count real tokens: `q_pos` already excludes the left padding.
        n_complete = mx.maximum(q_pos + 1, 0) // self.compress_ratio
        visible = mx.arange(n_blocks)[None, None, :] < n_complete[..., None]
        scores = mx.where(visible, scores, -mx.inf)

        k = min(self.block_topk, n_blocks)
        top = mx.stop_gradient(
            mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
        )  # (B, S, k)
        picked = mx.take_along_axis(visible, top, axis=-1)

        # remap block -> tokens; the buffer's trailing partial block belongs to no
        # candidate block, so it is not selectable on its own
        if pad is None:
            keep_block = mx.put_along_axis(
                mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_),
                mx.where(picked, top, n_blocks),
                mx.array(True),
                axis=-1,
            )[..., :n_blocks]
            keep = mx.repeat(keep_block, self.compress_ratio, axis=-1)
            rest = kv_len - n_blocks * self.compress_ratio
            if rest:
                keep = mx.concatenate(
                    [keep, mx.zeros((B, S, rest), dtype=mx.bool_)], axis=-1
                )
        else:
            # One block grid per row, so a repeat no longer expands it: scatter
            # the winners straight on the token axis.
            tok = (
                pad[:, None, None, None]
                + top[..., None] * self.compress_ratio
                + mx.arange(self.compress_ratio)[None, None, None, :]
            ).reshape(B, S, -1)
            flag = mx.broadcast_to(
                picked[..., None], (*picked.shape, self.compress_ratio)
            ).reshape(B, S, -1) & (tok < kv_len)
            keep = mx.put_along_axis(
                mx.zeros((B, S, kv_len + 1), dtype=mx.bool_),
                mx.where(flag, tok, kv_len),
                flag,
                axis=-1,
            )[..., :kv_len]

        # The reference appends the tail of each query's own visible list, i.e. the
        # partial block it sits in. Without it a query whose past holds fewer than
        # compress_ratio tokens gets an all-masked row, which softmax turns into a
        # uniform average over every key, future ones included.
        own_start = n_complete * self.compress_ratio
        if pad is not None:
            own_start = own_start + pad[:, None]
        tokens = mx.arange(kv_len)
        # Physical column of each query in the shared buffer.
        kv_pos = mx.arange(kv_len - S, kv_len)
        own = (tokens[None, None, :] >= own_start[..., None]) & (
            tokens[None, None, :] <= kv_pos[None, :, None]
        )
        return (keep | own)[:, None]  # (B, 1, S, kv_len)


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        # q_proj also carries the output gate: n_heads * head_dim * 2
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)
        self.kv_fused = False
        self.compiled_output = None

    def fuse_quantized(self):
        """Fuse packed MXFP4 projections that share an input."""
        return self.fuse_quantized_kv()

    def fuse_quantized_kv(self):
        """Fuse packed MXFP4 key and value projections."""
        if self.kv_fused:
            return False
        key, value = self.k_proj, self.v_proj
        if not isinstance(key, nn.QuantizedLinear) or not isinstance(
            value, nn.QuantizedLinear
        ):
            return False
        if key.group_size != value.group_size or key.bits != value.bits:
            return False
        if key.mode != value.mode:
            return False

        key.weight = mx.concatenate([key["weight"], value["weight"]], axis=0)
        key.scales = mx.concatenate([key["scales"], value["scales"]], axis=0)
        if "biases" in key:
            key.biases = mx.concatenate([key["biases"], value["biases"]], axis=0)
        mx.eval(key["weight"], key["scales"])
        self.v_proj = None
        self.kv_fused = True
        return True

    def _output(self, out: mx.array, gate: mx.array) -> mx.array:
        return self.o_proj(out * mx.sigmoid(gate))

    def compile_decode(self):
        self.compiled_output = mx.compile(self._output)

    def __call__(self, x, rope, mask, cache, idx_cache) -> mx.array:
        B, S, _ = x.shape
        offset = cache.offset if cache is not None else 0

        sparse = self.indexer(x, rope, idx_cache, offset, _left_padding(cache))

        q_all = self.q_proj(x)
        if self.kv_fused:
            kv_flat = self.k_proj(x)
        else:
            kv_flat = None
        q, gate = mx.split(q_all.reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        if kv_flat is None:
            k_flat, v_flat = self.k_proj(x), self.v_proj(x)
        else:
            k_flat, v_flat = mx.split(kv_flat, 2, axis=-1)
        k = self.k_norm(k_flat.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v_flat.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = rope(_positions(offset, S))
        cos, sin = cos[:, None], sin[:, None]
        q, k = _rope_partial(q, cos, sin), _rope_partial(k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if sparse is not None:
            # `sparse` is a boolean keep mask. Keep the combination boolean:
            # adding it drops the causality held by the "causal" string.
            if mask is None:
                mask = sparse
            elif isinstance(mask, str):  # "causal"
                kv_len = k.shape[2]
                rinds = mx.arange(kv_len)
                linds = mx.arange(kv_len - S, kv_len)[:, None]
                mask = (linds >= rinds) & sparse
            elif mask.dtype == mx.bool_:
                mask = mask & sparse
            else:
                neg = mx.finfo(mask.dtype).min
                mask = mask + mx.where(sparse, mx.array(0, mask.dtype), neg)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        if self.compiled_output is not None and B == 1 and S == 1:
            return self.compiled_output(out, gate)
        return self._output(out, gate)


# ------------------------------------------------------------------- gated deltanet


def _gdn_output(
    out: mx.array,
    z: mx.array,
    norm_weight: mx.array,
    weight: mx.array,
    scales: mx.array,
) -> mx.array:
    normed = mx.fast.rms_norm(out, norm_weight, 1e-6)
    gate = mx.sigmoid(z.astype(mx.float32))
    hidden = (gate * normed.astype(mx.float32)).astype(out.dtype)
    return mx.quantized_matmul(
        hidden.reshape(*out.shape[:-2], -1),
        weight,
        scales=scales,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )


_compiled_gdn_output = mx.compile(_gdn_output)


class GatedDeltaNet(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_v = args.linear_num_value_heads
        self.n_k = args.linear_num_key_heads
        self.dk = args.linear_key_head_dim
        self.dv = args.linear_value_head_dim
        self.key_dim = self.dk * self.n_k
        self.value_dim = self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.packed_t1 = bool(args.gdn_packed_t1)
        d = args.hidden_size

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        if args.prefused_gdn:
            projection_dim = self.conv_dim + self.value_dim + 2 * self.n_v
            self.in_proj_qkv = nn.Linear(d, projection_dim, bias=False)
            self.in_proj_z = None
            self.in_proj_b = None
            self.in_proj_a = None
            self.qkvz_fused = True
            self.ba_fused = True
            self.all_fused = True
            self.fuse_all_projections = False
        else:
            # unlike qwen3-next, the projections are split
            self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
            self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
            self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
            self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
            self.qkvz_fused = False
            self.ba_fused = False
            self.all_fused = False
            self.fuse_all_projections = args.fuse_gdn_all
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(
            self.dv, eps=args.rms_norm_eps, activation=args.output_gate_type
        )
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)
        self.compiled_output = None

    def _fuse_pair(self, first: str, second: str) -> bool:
        first_module, second_module = getattr(self, first), getattr(self, second)
        if not isinstance(first_module, nn.QuantizedLinear) or not isinstance(
            second_module, nn.QuantizedLinear
        ):
            return False
        if first_module.group_size != second_module.group_size:
            return False
        if first_module.bits != second_module.bits:
            return False
        if first_module.mode != second_module.mode:
            return False

        first_module.weight = mx.concatenate(
            [first_module["weight"], second_module["weight"]], axis=0
        )
        first_module.scales = mx.concatenate(
            [first_module["scales"], second_module["scales"]], axis=0
        )
        if "biases" in first_module:
            first_module.biases = mx.concatenate(
                [first_module["biases"], second_module["biases"]], axis=0
            )
        mx.eval(first_module["weight"], first_module["scales"])
        setattr(self, second, None)
        return True

    def fuse_quantized(self):
        """Fuse packed MXFP4 projections that share an input."""
        count = 0
        if not self.qkvz_fused and self._fuse_pair("in_proj_qkv", "in_proj_z"):
            self.qkvz_fused = True
            count += 1
        if not self.ba_fused and self._fuse_pair("in_proj_b", "in_proj_a"):
            self.ba_fused = True
            count += 1
        if (
            self.fuse_all_projections
            and self.qkvz_fused
            and self.ba_fused
            and not self.all_fused
            and self._fuse_pair("in_proj_qkv", "in_proj_b")
        ):
            self.all_fused = True
            count += 1
        return count

    def _output(self, out: mx.array, z: mx.array) -> mx.array:
        return self.out_proj(self.norm(out, z).reshape(*out.shape[:-2], -1))

    def compile_decode(self):
        self.compiled_output = True

    def _compiled_output(self, out: mx.array, z: mx.array) -> mx.array:
        return _compiled_gdn_output(
            out,
            z,
            self.norm.weight,
            self.out_proj.weight,
            self.out_proj.scales,
        )

    def __call__(self, x, mask, cache) -> mx.array:
        B, S, _ = x.shape
        projection = self.in_proj_qkv(x)
        if self.all_fused:
            qkvz, ba = mx.split(
                projection,
                [self.conv_dim + self.value_dim],
                axis=-1,
            )
            mixed_qkv, z_flat = mx.split(qkvz, [self.conv_dim], axis=-1)
            b, a = mx.split(ba, 2, axis=-1)
        else:
            qkvz = projection
            if self.qkvz_fused:
                mixed_qkv, z_flat = mx.split(qkvz, [self.conv_dim], axis=-1)
            else:
                mixed_qkv = qkvz
                z_flat = self.in_proj_z(x)
            ba = self.in_proj_b(x)
            if self.ba_fused:
                b, a = mx.split(ba, 2, axis=-1)
            else:
                b, a = ba, self.in_proj_a(x)
        z = z_flat.reshape(B, S, self.n_v, self.dv)

        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                # A right padded batch ends each row at its own length.
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        # The reference l2norm adds eps to sum(q^2); rms_norm adds it to the
        # mean, so convert via normalize_qk to keep the eps semantics exact.
        q, k = normalize_qk(q, k, inv_scale=self.dk**-0.5, eps=1e-6)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
            allow_packed_t1=self.packed_t1,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        if self.compiled_output is not None and B == 1 and S == 1:
            return self._compiled_output(out, z)
        return self._output(out, z)


# ------------------------------------------------------------------------- MoE


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            fused_gate_up=args.prefused_moe,
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)
        self.compiled_forward = None

    def _forward(self, x: mx.array) -> mx.array:
        logits = self.gate(x.astype(mx.float32))
        idx = mx.stop_gradient(
            mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        if self.switch_mlp.weight_before_down:
            out = self.switch_mlp.weighted_sum(x, idx, w).astype(x.dtype)
        else:
            out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    def __call__(self, x: mx.array) -> mx.array:
        if self.compiled_forward is not None and x.shape[0] == 1 and x.shape[1] == 1:
            return self.compiled_forward(x)
        return self._forward(x)

    def compile_decode(self):
        self.compiled_forward = mx.compile(self._forward)

    def fuse_quantized(self):
        count = self.switch_mlp.fuse_gate_up()
        return count + self.shared_expert.fuse_quantized()


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.fused_gate_up = False

    def fuse_quantized(self):
        """Fuse two packed MXFP4 projections into one matmul."""
        if self.fused_gate_up:
            return False
        gate, up = self.gate_proj, self.up_proj
        if not isinstance(gate, nn.QuantizedLinear) or not isinstance(
            up, nn.QuantizedLinear
        ):
            return False
        if gate.group_size != up.group_size or gate.bits != up.bits:
            return False
        if gate.mode != up.mode:
            return False

        gate.weight = mx.concatenate([gate["weight"], up["weight"]], axis=0)
        gate.scales = mx.concatenate([gate["scales"], up["scales"]], axis=0)
        if "biases" in gate:
            gate.biases = mx.concatenate([gate["biases"], up["biases"]], axis=0)
        mx.eval(gate["weight"], gate["scales"])
        self.up_proj = None
        self.fused_gate_up = True
        return True

    def __call__(self, x):
        if self.fused_gate_up:
            gate, up = mx.split(self.gate_proj(x), 2, axis=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(nn.silu(gate) * up)


# ------------------------------------------------------ hyper-connections (residual)


class GatedResidual(nn.Module):
    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc = args.hc_count
        self.d = args.hidden_size
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_dim, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dim, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_dim, self.hc, bias=False) if use_combine else None
        )
        self.compiled_forward = None

    def _forward(self, hyper: mx.array):
        normed = self.hc_norm(hyper)
        w = nn.silu(self.input_mix_weight_down(normed) / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper, inject

    def __call__(self, hyper: mx.array):
        if (
            self.compiled_forward is not None
            and hyper.shape[0] == 1
            and hyper.shape[1] == 1
        ):
            return self.compiled_forward(hyper)
        return self._forward(hyper)

    def compile_decode(self):
        self.compiled_forward = mx.compile(self._forward)


def _inject_block(x: mx.array, hyper: mx.array, inject: mx.array) -> mx.array:
    return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


_compiled_inject_block = mx.compile(_inject_block)


# -------------------------------------------------------------- n-gram / PLE


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class NGramEmbedding(nn.Module):
    """N-gram hash table, sharded into `split_ngram_parts` pieces.

    ~51B parameters: a dense lookup is never performed. Indices are sorted by shard
    on the host side, as in the llama.cpp implementation.
    """

    def __init__(self, args: TextArgs, embed_dim: int, ple_layer_index: int = 0):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = embed_dim // self.ngram_heads

        sizes, offsets, total = [], [], 0
        for h in range(self.ngram_heads):
            g = ple_layer_index * self.ngram_heads + h
            s = _nth_prime_after(args.ngram_vocab_size_base - 1, g + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        self.head_vocab_sizes = sizes

        div = args.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / div) * div
        self.n_shards = args.split_ngram_parts
        self.rows_per_shard = math.ceil(padded / self.n_shards)
        self.ngram_embedding = _ShardedEmbedding(
            self.n_shards,
            self.rows_per_shard,
            head_dim,
            args.ple_backend,
        )
        self.hash_backend = "host"
        if args.ple_backend:
            self.hash_backend = args.ple_backend.get("hash_backend", "host")
        self._ticket = None

        # buffers taken as-is from the checkpoint
        mults = []
        max_long = (1 << 63) - 1
        half = max(1, (max_long // max(args.vocab_size, 1)) // 2)
        base_seed = args.seed + _PRIME_1 * ple_layer_index
        for i in range(self.ngram_size):
            mults.append(
                2 * (_splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % half) + 1
            )
        # Public attributes: only there to absorb the checkpoint tensors. They live
        # in parameters(), so an astype(float16) would destroy them; the values
        # actually used live in the `_`-prefixed copies, outside parameters() and
        # rebuilt identically from the config.
        self.layer_multipliers = mx.array(mults, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self._mults = mx.array(mults, dtype=mx.int64)
        self._sizes = mx.array(sizes, dtype=mx.int64)
        self._offsets = mx.array(offsets, dtype=mx.int64)

    def _gid(self, ids: mx.array, prev_context: mx.array):
        backend = self.hash_backend
        if backend == "auto":
            backend = "mlx" if ids.shape[0] == 1 else "host"
        if backend == "mlx":
            return self._mlx_gids(ids, prev_context)
        return self._host_gids(ids, prev_context)

    def _shift_right(self, ids: mx.array, shift: int) -> mx.array:
        """Shift right by `shift`, without crossing an EOS boundary."""
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = mx.arange(T)
        eos_pos = mx.where(ids == self.eos_token_id, pos, -1)
        prev_incl = mx.cummax(eos_pos, axis=1)
        prev = mx.concatenate(
            [mx.full((B, 1), -1, dtype=prev_incl.dtype), prev_incl[:, :-1]], axis=1
        )
        in_segment = pos[None] - (prev + 1)
        src = pos - shift
        gathered = mx.take_along_axis(
            ids, mx.broadcast_to(mx.maximum(src, 0)[None], (B, T)), axis=1
        )
        ok = (in_segment >= shift) & (src[None] >= 0)
        return mx.where(ok, gathered, self.eos_token_id)

    def _host_gids(self, ids: mx.array, prev_context: mx.array) -> np.ndarray:
        n_new = ids.shape[1]
        history = np.concatenate(
            [
                np.asarray(prev_context.tolist(), dtype=np.int64),
                np.asarray(ids.tolist(), dtype=np.int64),
            ],
            axis=1,
        )
        batch, length = history.shape
        pos = np.arange(length)
        eos_pos = np.where(history == self.eos_token_id, pos[None], -1)
        prev_incl = np.maximum.accumulate(eos_pos, axis=1)
        prev = np.concatenate(
            [np.full((batch, 1), -1, dtype=np.int64), prev_incl[:, :-1]], axis=1
        )
        shifted = []
        for shift in range(self.ngram_size):
            src = pos[None] - shift
            gathered = np.take_along_axis(
                history, np.broadcast_to(np.maximum(src, 0), (batch, length)), axis=1
            )
            ok = ((pos[None] - prev - 1) >= shift) & (src >= 0)
            shifted.append(np.where(ok, gathered, self.eos_token_id))

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * int(self._mults[0].item())
            for p in range(1, ngram):
                multiplier = int(self._mults[p].item())
                mixed = np.bitwise_xor(mixed, shifted[p] * multiplier)
            sizes = self._sizes[lo:hi].tolist()
            offsets = self._offsets[lo:hi].tolist()
            gids = [mixed % size + offset for size, offset in zip(sizes, offsets)]
            blocks.append(np.stack(gids, axis=-1))

        gid = np.concatenate(blocks, axis=-1)[:, -n_new:]
        return gid

    def _mlx_gids(self, ids: mx.array, prev_context: mx.array) -> mx.array:
        n_new = ids.shape[1]
        history = mx.concatenate([prev_context, ids], axis=1)
        shifted = [
            self._shift_right(history, shift) for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self._mults[0]
            for p in range(1, ngram):
                mixed = mixed ^ (shifted[p] * self._mults[p])
            gids = [mixed % self._sizes[i] + self._offsets[i] for i in range(lo, hi)]
            blocks.append(mx.stack(gids, axis=-1))
        return mx.concatenate(blocks, axis=-1)[:, -n_new:]

    def __call__(self, ids: mx.array, prev_context: mx.array) -> mx.array:
        gid = self._gid(ids, prev_context)
        return self.ngram_embedding(gid).reshape(*gid.shape[:2], -1)

    def issue(self, ids: mx.array, prev_context: mx.array):
        gid = self._gid(ids, prev_context)
        self._ticket = (self.ngram_embedding.issue(gid), gid.shape[:2])

    def collect(self, ids: mx.array, prev_context: mx.array) -> mx.array:
        if self._ticket is None:
            return self(ids, prev_context)
        ticket, shape = self._ticket
        self._ticket = None
        return self.ngram_embedding.collect(ticket).reshape(*shape, -1)


_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.")
_PER_EXPERT_RE = re.compile(
    r"(.*)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight$"
)


class _ShardedEmbedding(nn.Module):
    """N embedding tables concatenated logically, addressed by global index."""

    def __init__(
        self,
        n_shards: int,
        rows: int,
        dim: int,
        backend: dict | None = None,
    ):
        super().__init__()
        self.n_shards = n_shards
        self.rows = rows
        self.dim = dim
        self.backend = backend
        if backend is None:
            # a list, so the keys are `shards.{i}.weight`: the layout mlx-vlm writes
            # and `nn.quantize` walks; `sanitize` maps the HF `shard_{i}` onto it
            self.shards = [nn.Embedding(rows, dim) for _ in range(n_shards)]
            self.external = None
        else:
            self.shards = []
            self.external = ExternalPleTable(
                backend["root"],
                n_shards=n_shards,
                rows_per_shard=backend.get("rows_per_shard", rows),
                max_inflight=backend.get("max_inflight", 32),
                cache_rows=backend.get("cache_rows", 0),
                prefetch=backend.get("prefetch", False),
                read_backend=backend.get("read_backend", "pread"),
            )
            if self.external.rows_per_shard != rows:
                self.close()
                raise ValueError(
                    f"PLE backend has {self.external.rows_per_shard} rows/shard, "
                    f"expected {rows}"
                )
            if self.external.values_per_row != dim:
                self.close()
                raise ValueError(
                    f"PLE backend has {self.external.values_per_row} values/row, "
                    f"expected {dim}"
                )

    def close(self):
        if self.external is not None:
            self.external.close()

    def clear_cache(self):
        if self.external is not None:
            self.external.clear_cache()

    def issue(self, gid):
        if self.external is None:
            return None
        return self.external.issue(gid)

    def collect(self, ticket):
        if self.external is None:
            raise RuntimeError("resident PLE table has no external ticket")
        return self.external.collect(ticket)

    def __del__(self):
        self.close()

    def __call__(self, gid: mx.array | np.ndarray) -> mx.array:
        if self.external is not None:
            return self.external.gather(gid)
        # which shards are actually touched: decided host-side, as llama.cpp does
        flat_indices = gid.reshape(-1).tolist()
        shard_indices = np.fromiter(
            (i // self.rows for i in flat_indices),
            dtype=np.int64,
            count=len(flat_indices),
        )
        row_indices = np.fromiter(
            (i % self.rows for i in flat_indices),
            dtype=np.int64,
            count=len(flat_indices),
        )
        touched = np.unique(shard_indices)
        out = mx.zeros((len(flat_indices), self.dim), dtype=mx.float32)
        for s in touched.tolist():
            positions = np.nonzero(shard_indices == s)[0]
            sel = mx.array(positions)
            rows = mx.array(row_indices[positions])
            emb = self.shards[s](rows)
            out = mx.put_along_axis(out, sel[:, None], emb.astype(mx.float32), axis=0)
        return out.reshape(*gid.shape, self.dim)


class PLELayer(nn.Module):
    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.d = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.d * self.hc
        self.ple_embedding = NGramEmbedding(args, args.ple_embed_dim, ple_layer_index)
        k = args.ple_conv_kernel_size
        self.dilation = args.ngram_size
        self.short_conv_state_len = (k - 1) * self.dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.d, bias=False)
        self.norm_key = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_query = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_dim,
            hc_dim,
            kernel_size=k,
            groups=hc_dim,
            dilation=self.dilation,
            bias=False,
        )

    def _short_conv(self, x: mx.array, cache) -> mx.array:
        S = x.shape[1]
        n = self.short_conv_state_len
        state = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            if cache.lengths is not None:
                # A right padded batch ends each row at its own length.
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n))[..., None]
                cache[2] = mx.take_along_axis(full, positions, axis=1)
            else:
                cache[2] = mx.contiguous(full[:, -n:, :])
        return nn.silu(self.conv1d(full[:, -(n + S) :, :]))

    def issue(self, ids: mx.array, prev_ctx: mx.array):
        self.ple_embedding.issue(ids, prev_ctx)

    def __call__(
        self, hidden: mx.array, ids: mx.array, prev_ctx: mx.array, cache
    ) -> mx.array:
        emb = self.ple_embedding.collect(ids, prev_ctx).astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc, self.d)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc, self.d)

        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*gated.shape[:-2], -1)
        return gated + self._short_conv(self.norm_conv(gated), cache)


# ------------------------------------------------------------------- decoder / model


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        ple_idx = (
            args.ple_layer_ids.index(layer_idx + 1)
            if (layer_idx + 1) in args.ple_layer_ids
            else None
        )
        self.ple = PLELayer(args, ple_idx) if ple_idx is not None else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)
        self.compile_injection = args.compile_injection

    def __call__(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        if self.ple is not None:
            h = h + self.ple(h, ids, prev_ctx, cache)

        x, hyper, inject = self.attn_hyper_connection(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
        else:
            x = self.self_attn(x, rope, mask, cache, idx_cache)
        if self.compile_injection and x.shape[0] == 1 and x.shape[1] == 1:
            h = _compiled_inject_block(x, hyper, inject)
        else:
            h = _inject_block(x, hyper, inject)

        x, hyper, inject = self.mlp_hyper_connection(h)
        x = self.mlp(x)
        if self.compile_injection and x.shape[0] == 1 and x.shape[1] == 1:
            return _compiled_inject_block(x, hyper, inject)
        return _inject_block(x, hyper, inject)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        # no final `norm` in this model: this mixer carries it
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope = RotaryEmbedding(rotary_dim, args.rope_theta)
        self.ple_layers = [
            i for i in range(args.num_hidden_layers) if (i + 1) in args.ple_layer_ids
        ]
        self.compile_moe = args.compile_moe
        self.compile_residual = args.compile_residual
        self.compile_injection = args.compile_injection
        self.compile_attention_output = args.compile_attention_output
        self.compile_gdn_output = args.compile_gdn_output
        self.gdn_packed_t1_max_context = int(args.gdn_packed_t1_max_context or 0)
        ple_backend = args.ple_backend or {}
        self.ple_base_cache_rows = int(ple_backend.get("cache_rows", 0))
        self.ple_long_cache_rows = int(ple_backend.get("long_context_cache_rows", 0))
        self.ple_long_context_threshold = int(
            ple_backend.get("long_context_threshold", 0)
        )
        self._ple_context_tokens = 0

    def close_ple(self):
        for i in self.ple_layers:
            self.layers[i].ple.ple_embedding.ngram_embedding.close()

    def set_ple_cache_capacity(self, cache_rows: int):
        for i in self.ple_layers:
            embedding = self.layers[i].ple.ple_embedding.ngram_embedding
            if embedding.external is not None:
                embedding.external.set_cache_capacity(cache_rows)

    def prepare_prompt_tokens(self, total_tokens: int):
        if (
            self.ple_long_cache_rows > self.ple_base_cache_rows
            and self.ple_long_context_threshold
            and total_tokens > self.ple_long_context_threshold
        ):
            self.set_ple_cache_capacity(self.ple_long_cache_rows)

    def clear_ple_cache(self, reset_capacity: bool = False):
        for i in self.ple_layers:
            embedding = self.layers[i].ple.ple_embedding.ngram_embedding
            if embedding.external is None:
                continue
            if reset_capacity:
                embedding.external.set_cache_capacity(self.ple_base_cache_rows)
            else:
                embedding.external.clear_cache()

    def reset_prompt_cache(self, cache):
        for layer_cache in cache:
            layer_cache.reset()
        self._ple_context_tokens = 0
        if self.ple_base_cache_rows:
            self.clear_ple_cache(reset_capacity=True)

    def fuse_quantized(self):
        count = sum(layer.mlp.fuse_quantized() for layer in self.layers)
        count += sum(
            layer.self_attn.fuse_quantized()
            for layer in self.layers
            if layer.layer_type == "full_attention"
        )
        count += sum(
            layer.linear_attn.fuse_quantized()
            for layer in self.layers
            if layer.layer_type == "linear_attention"
        )
        mx.clear_cache()
        return count

    def compile_decode(self):
        if self.compile_moe:
            for layer in self.layers:
                layer.mlp.compile_decode()
        if self.compile_residual:
            for layer in self.layers:
                layer.attn_hyper_connection.compile_decode()
                layer.mlp_hyper_connection.compile_decode()
            self.hyper_connection_mixer.compile_decode()
        if self.compile_attention_output:
            for layer in self.layers:
                if layer.layer_type == "full_attention":
                    layer.self_attn.compile_decode()
        if self.compile_gdn_output:
            for layer in self.layers:
                if layer.layer_type == "linear_attention":
                    layer.linear_attn.compile_decode()

    def __call__(self, ids: mx.array, cache=None, input_embeddings=None):
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        full_idx = [
            i for i, l in enumerate(self.layers) if l.layer_type == "full_attention"
        ]
        lin_idx = [
            i for i, l in enumerate(self.layers) if l.layer_type == "linear_attention"
        ]
        mask = create_attention_mask(h, cache[full_idx[0]] if full_idx else None)
        # The deltanet reads the tokens in order, so it needs the padded ones
        # zeroed out. Its cache holds the padding, one value per row.
        conv_mask = create_ssm_mask(h, cache[lin_idx[0]]) if lin_idx else None

        prev_ctx = None
        # The n-gram hash reads `ids` in order, so it holds for the right padded
        # prompt of a batch, but not for one that is already left padded.
        if self.ple_layers:
            ctx_len = self.args.ngram_size - 1
            eos = self.args.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            pc = cache[self.ple_layers[0]]
            prev = pc[3] if pc is not None else None
            prev_ctx = (
                prev
                if prev is not None
                else mx.full((ids.shape[0], ctx_len), eos, ids.dtype)
            )
            context_tokens = self._ple_context_tokens + ids.shape[1]
            if (
                self.ple_long_cache_rows > self.ple_base_cache_rows
                and self.ple_long_context_threshold
                and context_tokens > self.ple_long_context_threshold
            ):
                self.set_ple_cache_capacity(self.ple_long_cache_rows)
            self._ple_context_tokens = context_tokens
            if pc is not None:
                history = mx.concatenate([prev_ctx, ids], axis=1)
                if pc.lengths is not None:
                    ends = mx.clip(pc.lengths, 0, ids.shape[1])
                    pc[3] = mx.take_along_axis(
                        history, ends[:, None] + mx.arange(ctx_len), axis=1
                    )
                else:
                    pc[3] = history[:, -ctx_len:]
            ple_embedding = self.layers[self.ple_layers[0]].ple.ple_embedding
            external = ple_embedding.ngram_embedding.external
            if external is not None and external.prefetch:
                ple_embedding.issue(ids, prev_ctx)

        h = mx.tile(h, (1, 1, self.hc))
        if self.gdn_packed_t1_max_context:
            packed_t1 = self._ple_context_tokens <= self.gdn_packed_t1_max_context
            for layer in self.layers:
                if layer.layer_type == "linear_attention":
                    layer.linear_attn.packed_t1 = packed_t1
        for layer, c in zip(self.layers, cache):
            idx_c = c.indexer if (c is not None and hasattr(c, "indexer")) else None
            h = layer(h, self.rope, mask, conv_mask, c, idx_c, ids, prev_ctx)
        return self.hyper_connection_mixer(h)


class _IndexerCache(_BaseCache):
    """Holds the indexer raw keys (one per token, not pooled)."""

    def __init__(self):
        self.keys = None

    def update(self, k: mx.array) -> mx.array:
        self.keys = k if self.keys is None else mx.concatenate([self.keys, k], axis=1)
        return self.keys

    def trim(self, size: int):
        if self.keys is not None:
            self.keys = self.keys[:, :size]

    @property
    def state(self):
        # `None` is not serializable, so an empty array stands for it.
        return mx.array([]) if self.keys is None else self.keys

    @state.setter
    def state(self, v):
        self.keys = v if v.size > 0 else None


class _AttnCache(KVCache):
    def __init__(self):
        super().__init__()
        self.indexer = _IndexerCache()

    def trim(self, n):
        # `KVCache.trim` only moves the offset, but the indexer keys are exact.
        n = super().trim(n)
        self.indexer.trim(self.offset)
        return n

    @property
    def state(self):
        return (*super().state, self.indexer.state)

    @state.setter
    def state(self, v):
        *kv, self.indexer.state = v
        KVCache.state.fset(self, tuple(kv))

    @classmethod
    def merge(cls, caches):
        return _BatchAttnCache.merge(caches)

    def reset(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self.indexer.keys = None


class _BatchAttnCache(BatchKVCache):
    """Batched `_AttnCache`. The indexer keys follow the KV padding exactly.

    Credit: mirrors BatchQSAKVCache from Blaizzy/mlx-vlm#2028 (MIT).
    """

    def __init__(self, left_padding):
        super().__init__(left_padding)
        self.indexer = _IndexerCache()

    def reset(self):
        self.keys = None
        self.values = None
        self.left_padding = mx.zeros_like(self.left_padding)
        self.offset = mx.zeros_like(self.offset)
        self._idx = 0
        self._right_padding = None
        self.indexer.keys = None

    def finalize(self):
        # The base class rolls a right padded prefill into left padding. The
        # indexer keys share those columns, so they take the same roll.
        padding = self._right_padding
        super().finalize()
        if padding is not None and self.indexer.keys is not None:
            self.indexer.keys = dynamic_roll(self.indexer.keys, padding, axis=1)

    def trim(self, n):
        n = super().trim(n)
        self.indexer.trim(self._idx)
        return n

    def filter(self, batch_indices):
        min_pad = self.left_padding[batch_indices].min().item()
        super().filter(batch_indices)
        if self.indexer.keys is not None:
            keys = self.indexer.keys[batch_indices]
            self.indexer.keys = keys[:, min_pad:] if min_pad else keys

    def extend(self, other):
        keys, other_keys = self.indexer.keys, other.indexer.keys
        idx, other_idx = self._idx, other._idx
        super().extend(other)
        if keys is None and other_keys is None:
            return
        sample = keys if keys is not None else other_keys
        target = max(idx, other_idx)
        rows = self.offset.shape[0] - other.offset.shape[0]

        # Right-justify both sides on the shared index, like `BatchKVCache`.
        def pad(k, n_rows, used):
            if k is None:
                k = mx.zeros((n_rows, 0, sample.shape[-1]), dtype=sample.dtype)
            return mx.pad(k[:, :used], [(0, 0), (target - used, 0), (0, 0)])

        self.indexer.keys = mx.concatenate(
            [pad(keys, rows, idx), pad(other_keys, other.offset.shape[0], other_idx)],
            axis=0,
        )

    def extract(self, idx):
        cache = _AttnCache()
        pad = self.left_padding[idx].item()
        if self.keys is not None:
            cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, pad : self._idx])
            cache.values = mx.contiguous(self.values[idx : idx + 1, :, pad : self._idx])
            cache.offset = cache.keys.shape[2]
        if self.indexer.keys is not None:
            cache.indexer.keys = mx.contiguous(
                self.indexer.keys[idx : idx + 1, pad : self._idx]
            )
        return cache

    @classmethod
    def merge(cls, caches):
        # `BatchKVCache.merge` names its own class on the all-empty path, which
        # drops the indexer.
        if max(c.size() for c in caches) == 0:
            return cls([0] * len(caches))
        out = super().merge(caches)
        rows = [c.indexer.keys for c in caches]
        sample = next((k for k in rows if k is not None), None)
        if sample is None:
            return out
        out.indexer.keys = mx.concatenate(
            [
                (
                    mx.zeros((1, out._idx, sample.shape[-1]), dtype=sample.dtype)
                    if k is None
                    else mx.pad(
                        k[:, : c.offset],
                        [(0, 0), (out._idx - c.offset, 0), (0, 0)],
                    )
                )
                for k, c in zip(rows, caches)
            ],
            axis=0,
        )
        return out

    @property
    def state(self):
        return (*super().state, self.indexer.state)

    @state.setter
    def state(self, v):
        *kv, self.indexer.state = v
        BatchKVCache.state.fset(self, tuple(kv))


class _LayerCache(ArraysCache):
    """`ArraysCache` whose slots can stay unused on every row.

    The PLE slots are only filled by the PLE layers, and `ArraysCache` raises
    `StopIteration` on a slot that is `None` everywhere.
    """

    @classmethod
    def merge(cls, caches):
        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)
        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            return cache
        for e in range(n_state):
            c_init = next((c[e] for c in caches if c[e] is not None), None)
            if c_init is None:
                continue
            shape = list(c_init.shape)
            shape[0] = B
            cache[e] = mx.zeros(shape, c_init.dtype)
            for i in range(B):
                if caches[i][e] is not None:
                    cache[e][i : i + 1] = caches[i][e]
        return cache

    def extract(self, idx):
        cache = _LayerCache(len(self.cache))
        cache.cache = [None if c is None else c[idx : idx + 1] for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        # An empty merge leaves a zero `left_padding`, which shadows `lengths`
        # in `make_mask` and lets the right padding into the deltanet.
        self.left_padding = None
        super().prepare(lengths=lengths, **kwargs)

    def reset(self):
        self.cache = [None] * len(self.cache)
        self.left_padding = None
        self.lengths = None


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args.text)
        self.compile_moe = args.text.compile_moe
        self.compile_residual = args.text.compile_residual
        self.compile_injection = args.text.compile_injection
        self.compile_attention_output = args.text.compile_attention_output
        self.compile_gdn_output = args.text.compile_gdn_output
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.text.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def close(self):
        self.model.close_ple()

    def clear_ple_cache(self):
        self.model.clear_ple_cache()

    def reset_prompt_cache(self, cache):
        self.model.reset_prompt_cache(cache)

    def prepare_prompt_tokens(self, total_tokens: int):
        self.model.prepare_prompt_tokens(total_tokens)

    def fuse_quantized(self):
        return self.model.fuse_quantized()

    def compile_decode(self):
        """Compile the MoE block for single-token decode."""
        self.model.compile_decode()

    def make_cache(self):
        caches = []
        for _i, t in enumerate(self.args.text.layer_types):
            if t == "full_attention":
                caches.append(_AttnCache())
            else:
                # 0: deltanet conv, 1: ssm state, 2: PLE conv, 3: n-gram context
                caches.append(_LayerCache(4))
        return caches

    # Norms scaled by `1 + w` in the reference (RMSNorm above); `linear_attn.norm` is
    # RMSNormGated and scales by `w` alone, so it must stay out of this list.
    _FOLD_ONE = (
        "q_layernorm.weight",
        "k_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "hc_norm.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
    )

    def sanitize(self, weights):
        # Fold the reference's `1 +` into the weight here instead of at every call.
        # transformers saves the wrapper's text tower under `model.language_model.`,
        # the mlx-vlm converter preserves the nested `language_model.` spelling
        # (values verified still zero-centered there); both must fold. Only a
        # checkpoint saved from this file comes back flat as `model.*` with the
        # fold already applied, so it must not be folded twice.
        fold = any(
            k.startswith(("model.language_model.", "language_model.")) for k in weights
        )

        out = {}
        per_expert: dict = {}
        for k, v in weights.items():
            # Multi-token-prediction head and vision tower: not implemented by this
            # text-only port, and absent from the module tree -> drop them.
            if k.startswith(("mtp.", "model.mtp.")):
                continue
            if k.startswith(("model.visual.", "visual.", "vision_tower.")):
                continue

            # Prefix. The official checkpoint nests the text model under
            # `model.language_model.` while already-converted MLX checkpoints use
            # the flat `language_model.` form; both must land on `model.`.
            # `lm_head.weight` sits at the top level upstream and is left alone.
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            elif k.startswith("language_model."):
                k = k[len("language_model.") :]

            # N-gram table: HF names the shards `shard_{i}`, this file `shards.{i}`.
            k = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1.", k)
            if self.args.text.ple_backend and ".ngram_embedding.shards." in k:
                continue

            if self.args.text.prefused_moe and ".mlp.switch_mlp.gate_up_proj." in k:
                out[k.replace(".gate_up_proj.", ".gate_proj.")] = v
                continue

            # Experts. Upstream stacks them as `experts.gate_up_proj`
            # (E, 2 * moe_intermediate, hidden) and `experts.down_proj`
            # (E, hidden, moe_intermediate), i.e. already the (E, out, in) layout
            # SwitchGLU wants. The reference splits the *output* of the fused
            # projection with chunk(2, dim=-1), so gate is the first half of
            # axis -2 of the weight and up the second.
            if k.endswith("mlp.experts.gate_up_proj"):
                base = k[: -len("experts.gate_up_proj")]
                mid = v.shape[-2] // 2
                out[base + "switch_mlp.gate_proj.weight"] = v[..., :mid, :]
                out[base + "switch_mlp.up_proj.weight"] = v[..., mid:, :]
                continue
            if k.endswith("mlp.experts.down_proj"):
                base = k[: -len("experts.down_proj")]
                out[base + "switch_mlp.down_proj.weight"] = v
                continue

            # Per-expert layout, as saved by transformers save_pretrained
            # (`experts.{i}.gate_proj` etc. instead of one stacked tensor).
            m = _PER_EXPERT_RE.match(k)
            if m:
                base, idx, proj = m.group(1), int(m.group(2)), m.group(3)
                if proj == "gate_up_proj":
                    mid = v.shape[-2] // 2
                    slot = per_expert.setdefault(base, {}).setdefault(idx, {})
                    slot["gate_proj"] = v[..., :mid, :]
                    slot["up_proj"] = v[..., mid:, :]
                else:
                    per_expert.setdefault(base, {}).setdefault(idx, {})[proj] = v
                continue

            # (C, 1, K) torch -> (C, K, 1) mlx. Idempotent: an already converted
            # weight has shape[1] == kernel_size != 1.
            if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[1] == 1:
                v = v.transpose(0, 2, 1)

            if fold and k.endswith(self._FOLD_ONE):
                v = 1.0 + v

            out[k] = v

        for base, experts in per_expert.items():
            n = len(experts)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                out[f"{base}.mlp.switch_mlp.{proj}.weight"] = mx.stack(
                    [experts[i][proj] for i in range(n)], axis=0
                )
        return out

    @property
    def quant_predicate(self):
        # mlx-lm 0.31.3 calls predicates with two arguments, later versions with
        # three; the default keeps both conventions working
        def fn(path, module, _=None):
            # only the MoE router stays in full precision (norms and conv1d are
            # never quantized anyway)
            return not path.endswith("mlp.gate")

        return fn
