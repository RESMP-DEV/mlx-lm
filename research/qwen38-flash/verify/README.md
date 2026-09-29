# Logit-level parity verification for the qwen4_exp MLX port

Ground truth is the pure-PyTorch reference path in transformers 5.17.0
(`causal_conv1d` and `flash-linear-attention` deliberately NOT installed, so
GDN/conv run the reference kernels). The tiny random model exercises every
mechanism — GDN, QSA indexer + sparse core attention, Gated Residual, MoE,
PLE n-gram hashing with EOS resets — at a size where fp32 logits are directly
comparable end to end.

## Layout

- `gen_tiny_reference.py` — ref env (`.venv-ref`). Builds a tiny random
  `Qwen4ExpForConditionalGeneration`, saves an HF-format checkpoint to
  `tiny-qwen4exp/`, writes `prompts.json`, dumps `ref_logits.npz`.
- `dump_mlx_logits.py` — mlx env (`/Users/kearm/mlx/.venv`). Loads the same
  checkpoint through `mlx_lm.models.qwen4_exp` (config -> ModelArgs ->
  sanitize -> weights), dumps `mlx_logits.npz`.
- `compare.py` — either env. Reports per-prompt max/mean |diff|, argmax
  agreement, top-10 overlap; PASS below tolerance.
- `micro_gdn_ref.py` / `micro_gdn_mlx.py` / `micro_gdn_compare.py` — stage-level
  GDN bisect: identical `x_in` into layer 0's GDN on both frameworks, every
  intermediate compared (projections, conv, q/k norm, g/beta, kernel core,
  module out). `core_fromref` feeds the reference's exact kernel inputs through
  the MLX ops kernel, isolating kernel math from normalization.

## Prompts

- `dense` (10 tok): inside the dense-exact region for budget 8, ratio 4
  (threshold 11 = budget + ratio - 1) — must match reference exactly.
- `sparse` (24 tok, not a multiple of 4): beyond the budget, exercises QSA
  block selection plus the incomplete tail block.
- `eos_reset` (18 tok, EOS at 5 and 12): exercises PLE segment-relative
  position reset.

## Run

```bash
.venv-ref/bin/python research/qwen38-flash/verify/gen_tiny_reference.py
/Users/kearm/mlx/.venv/bin/python research/qwen38-flash/verify/dump_mlx_logits.py
.venv-ref/bin/python research/qwen38-flash/verify/compare.py
```

## Results (2026-09-28)

All variants PASS at fp32 rounding on both CPU and GPU paths:

| variant | max abs logits diff |
|---|---|
| tiny-qwen4exp (full, all mechanisms) | 1.04e-07 |
| tiny-gdn (3x GDN + 1x full) | ~1e-7 |
| tiny-attn (full-attention only) | ~1e-7 |
| tiny-ple / tiny-nople | ~1e-7 |
| tiny-alias (`qwen_sparse_attention` spelling) | ~1e-7 |

The GDN micro-bisect found one real bug: the port normalized q/k with
`mx.fast.rms_norm(..., eps=1e-6)` directly, but the reference `l2norm` adds
eps to sum(q^2) while rms_norm adds it to mean(q^2) — a dk-times eps mismatch.
On the tiny model mean(q^2) ~ eps, so the denominator was ~2x off (35% q_norm
error). Fixed by routing through `gated_delta.normalize_qk`, which converts
the eps (`eps * inv_scale**2`). After the fix every GDN stage matches to
<= 1e-7, the kernel core to 3.6e-11 (MLX sequential ops vs torch chunked —
mathematically equivalent, differing only in fp32 rounding).

## Notes

- layer_types spelling: the official config (and this generator) uses
  `full_attention`; transformers defaults emit `qwen_sparse_attention` for
  the same layer kind. The port dispatches on `full_attention` — the port
  normalizes both spellings.
- The reference checkpoint keeps the unfolded (zero-centered) norms; the
  port's sanitize folds the `1 +` on load, gated on the
  `model.language_model.` layout, so parity also proves the folding.
- Later milestones: same harness on the real 4-bit checkpoint is qualitative
  only (quantized reference unavailable); dense-vs-sparse MLX self-parity at
  <= 2051 tokens covers M1 instead.
