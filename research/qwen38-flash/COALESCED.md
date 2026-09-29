# Qwen3.8-Flash-Next (`qwen4_exp`) — coalesced research for the RESMP-DEV MLX port

Date: 2026-09-28. Method: five serialized GLM-5.3-FlashX research workers
(transformers, Qwen/Megatron, vLLM kernels, llama.cpp/GGUF, MLX ecosystem)
plus direct parent verification of the official `config.json` and README.
Raw worker outputs live beside this file (`w1`–`w5`); official snapshots in
`official_config.json` and `official_readme.md`.

## The model in one paragraph

Qwen3.8-Flash-Next (released 2026-08-26, `Qwen/Qwen3.8-Flash-Next`) is the
Qwen3-Next analogue of the Qwen4 family: a 125B-total / 6B-active hybrid MoE
whose dense full attention is replaced by trained sparse **QSA**, whose
residual stream becomes four **Gated-Residual** streams, and whose capacity
is scaled by a hashed **PLE n-gram embedding table** of 51.2B parameters at
layer 2 — plus a 1-layer **MTP** draft (4B). It ships as a VLM
(`Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp`) around a
`qwen4_exp_text` core. Context 262,144 native, 1M via YaRN (factor 4.0,
`mrope_section [11,11,10]`, `partial_rotary_factor 0.25`). Tokenizer is
`Qwen2Tokenizer`, padded vocab 248,320. License: Qwen Community 1.0
(attribution conditions above 100M MAU / $20M monthly revenue — matters if
we ever redistribute converted weights).

## Ground-truth topology (official config, verified)

- 48 layers, pattern 12 × [GDN, GDN, GDN, QSA]; `full_attention_interval 4`.
- GDN: 16 key / 48 value heads, dims 128, conv kernel 4, **sigmoid** output
  gate (qwen3_next used SiLU — Megatron made it configurable; vLLM ships
  sigmoid for qwen4_exp).
- Full-attention layers are QSA: core GQA 24 q / 2 kv heads, head_dim 256,
  partial RoPE 0.25, theta 1e7, mrope interleaved. Indexer: MQA 4 q / 1 kv
  head, head_dim 128, budget 2048, compress ratio 4 (→ top-512 blocks).
- MoE: 512 experts, top-10, expert intermediate 640, shared expert 640.
- GR/HC: `hc_count 4`, `hc_lowrank 320`, at every attention and MLP sublayer;
  final GR mixer replaces the final norm.
- PLE at layer id 2 (**0-based mapping** — trap), embed dim 2560, conv
  kernel 4, dilation 3 (9-token causal halo).
- MTP: 1 full-attention (QSA-backed) layer, hybrid, reuses step-0 QSA
  selections; SGLang serves NEXTN with 3 steps / 4 draft tokens.
- Hidden 2560. Params: 125B backbone (6B active) + 51.2B n-gram + 4B MTP.

## Mechanism mechanics

### QSA (Qwen Sparse Attention)

For query i, complete key blocks of r=4 tokens are average-pooled *before*
RoPE; the pooled key takes the block-start position. Indexer score
`I[i,b] = Σ_h ReLU(⟨q[i,h], k̄[b]⟩)/√d`; hard top-512 blocks; selected set
`S[i] = expand(B[i]) ∪ tokens(⌊i/r⌋·r .. i)` (local window). Core GQA runs
over S[i] only. **Exact dense equivalence for sequences ≤ 2051 tokens**
(2048 budget + tail): all visible complete blocks are selected. Trained
approximate beyond that. Reported 7.6× prefill / 4.9× decode at 1M context.
Quantized-KV checkpoints apply an extra Hadamard rotation (trap from
llama.cpp; the NVFP4/FP8 ckpts need it).

### Gated Residual (hyper-connections)

Per attention/MLP sublayer, over 4 banks of width 2560:

```
xn       = GroupRMSNorm(x)                    # zero-centered, per bank
mix      = sigmoid(W_up silu(W_down xn / n))  # rank-320 bottleneck
block_in = mean_j(mix_j * xn_j)
inject   = 2 * sigmoid(W_inject xn / n)
x_new_j  = x_j + inject_j * F(block_in)       # scalar-gated write to all banks
```

No separate input norms; final `GatedResidualOutputMixer` emits the 2560
hidden state for logits. The 4 banks are decode-time state and must flow
into the MTP combiner (per-stream — trap).

### PLE (hashed n-gram capacity layer, before layer 2's attention)

16 heads = `(ngram_size-1) × heads_per_ngram` = 2 × 8. Each head hashes the
bigram/trigram ending at t into its own prime-sized table just above 20M
rows (sum 320,001,446 rows) × 160 dim = 51.20B params ≈ 95.4 GiB BF16
(GGUF ships it transposed as `per_layer_token_embd.weight`
`[160, 320001536]`). Hashing is deterministic (per-head SplitMix64
multipliers, prime moduli); positions are segment-relative and **reset at
EOS / packed-sequence boundaries** (trap). Runtime:

```
e      = concat(E_h[hash_h(ngram_t)])
key    = GroupRMSNorm(Wk e);  value = Wv e
gate_j = signed_sqrt(⟨key_j, GroupRMSNorm(x)_j⟩ / √C)
update = sigmoid(gate_j)·value;  update += silu(DilatedDWConv1d(update))
bank  += update                       # writes all four banks
```

`split_ngram_parts=128` is checkpoint-shard concatenation only. The table
is input-side capacity ("amenable to offloading"): few rows touched per
token, so paging/quantization is the intended serving strategy. vLLM merged
PLE loading/offload (#54722, #54882) with NVFP4 packed lookup (#56273) and
disk offload (#54070) pending.

### GDN deltas vs qwen3_next

Geometry unchanged (state ≈ 817K elems/layer/seq: conv (10240,3) +
temporal (48,128,128), F32 conv/SSM state). Changes: sigmoid output gating,
GR replaces pre-norms, checkpoint key mapping must be re-verified.

### MTP

transformers ignores `mtp.*` entirely. vLLM implements `Qwen4ExpMTP`: a
standalone 1-layer full-attention (QSA) draft that fuses the target's
pre-final-mixer multi-stream hidden state with next-token embeddings via an
`fc`, then runs the layer and computes logits. Top-k indices selected at
draft step 0 are reused across steps (`set_skip_topk`). GGUF ships drafts
in separate `MTP/*.gguf` files under synthetic `blk.48.nextn.*` names;
`qwen4exp.nextn_shared_target_tensors=true` marks shared-target drafts
(omit `token_embd`/`output`). llama.cpp reportedly attends densely in the
draft layer — discrepancy vs QSA-backed MTP in vLLM/sglang; verify which is
reference-correct when implementing.

## Ecosystem landscape (what already exists)

| Stack | Status | Key reference |
|---|---|---|
| transformers | Shipped, PR #48337, release v5.17.0 | `src/transformers/models/qwen4_exp/` (modular file exists) |
| vLLM | **Merged**, PR #53896 (2026-08-31) | `vllm/models/qwen4_exp/` (nvidia/amd), registry `Qwen4ExpForCausalLM/ForConditionalGeneration/MTP`; config subclasses `Qwen3NextConfig`; BF16 main KV |
| llama.cpp | Merged (arch `qwen4exp`) + MTP PR #28243 | `per_layer_token_embd.weight`, `blk.48.nextn.*`, separate MTP ggufs |
| Megatron-LM | PR #7393 **open** | `qsa.py`, `per_layer_embedding.py`, `hyper_connection.py`; no MTP/inference caches |
| sglang | QSA runtime reference | `srt/layers/attention/qsa/qsa_indexer.py` (index width 2051), sparse backend; PRs 36664/39724 |
| **mlx-lm** | **Open PR #1788** (eauchs, 1 approval) | text-only `mlx_lm/models/qwen4_exp.py` + test; drops `mtp.*`/vision; generation + BatchGenerator parity evidence |
| **Blaizzy/mlx-vlm** | **Merged, most complete MLX impl** | `mlx_vlm/models/qwen4_exp/` + `speculative/drafters/qwen4_exp_mtp/`; MTP #2040, external PLE #2045, adaptive draft depth #2046, QSA cache/batch parity #2126, APC #2272 |
| jundot/omlx | Merged multimodal #3169 | oQ affine quant packs |
| ddalcu/mlx-serve | Merged YaRN + GDN/MTP perf (#323, #517) | external `ngram_table.bin` (32 GB safetensors-format PLE outside normal MLX loading) |

mlx-community HF conversions (all declare `qwen4_exp`): `4bit`, `4bit-mtp`,
`oQ6e-mtp`, `oQ8e-mtp`, `Uncensored-oQ5e/oQ6e-mtp`, `OptiQ-2bit` (mixed,
not truly 2-bit). All carry n-gram + indexer tensors; `-mtp` variants carry
draft weights. **Acceptance-checkpoint warning from PR #1788: pre-`ac83bb4`
conversions can have incompatible unbaked RMSNorm conventions** — pin the
acceptance artifact to a post-`ac83bb4` mlx-community conversion.

## Checkpoint / layout facts

- Registry wiring in mlx-lm is dynamic: `mlx_lm/utils.py:198-224` imports
  `mlx_lm.models.{model_type}`; `models/__init__.py` has no static map. A
  new `qwen4_exp.py` exporting `Model`/`ModelArgs` is the entire
  registration. `MODEL_REMAPPING` only needed to alias other type strings.
- PR #1788 defines model-local `_AttnCache/_BatchAttnCache/_IndexerCache/
  _LayerCache` instead of touching `mlx_lm/models/cache.py`.
- `sanitize()` must strip `mtp.*`/`vision` keys, stack MoE experts into
  `switch_mlp`, transpose conv weights, and unbake zero-centered norms
  (qwen3_next adds 1.0 to selected norms — verify qwen4_exp convention).
- mlx-lm `generate.py` supports only an external `draft_model`; native
  same-checkpoint MTP needs a drafter/verifier API or an extracted draft.
- MLX core (verified in our fork at `53aa45667`): `cumsum` (Metal scan
  kernels), `topk` (values), `argpartition` + `take_along_axis` (indices),
  `gather`, `put_along_axis` (PR #1788 uses it for PLE row sharding),
  `mx.fast.gated_delta_update` **with the exact Dk=128/Dv=128/Hk=16/Hv=48
  instantiation this model needs**, `mx.conv1d` grouped+dilated, SDPA with
  GQA. Missing: named chunked-cumsum (compose from cumsum + reshape) and a
  generic associative scan (not needed for a correct first port).

## Porting traps (consolidated from llama.cpp / ds4 / mistral.rs / workers)

1. EOS-based PLE hash reset (segment-relative positions; which EOS id —
   text-config 248044 vs chat 248046 — is unresolved).
2. 0-based `ple_layer_ids` mapping.
3. Per-stream (4-bank) input to the MTP combiner, pre-final-mixer hidden.
4. Dense vs QSA attention inside the MTP draft layer (llama.cpp vs vLLM).
5. Hadamard rotation on QSA KV when the checkpoint is quantized (NVFP4/FP8).
6. Atomic rollback of auxiliary caches (QSA index cache, PLE conv/hash
   history, GDN F32 state) on rejected speculative tokens.
7. Transposed PLE table layout in GGUF-derived artifacts.
8. Sigmoid (not SiLU) GDN output gating; checkpoint key remap vs qwen3_next.
9. Pre-`ac83bb4` MLX conversions may carry unbaked RMSNorm conventions.
10. mistral.rs left speculative/paged support explicitly unfinished — do
    not treat it as a reference for those paths.

## Recommended plan for our fork

**Strategy: don't port from transformers cold.** Start from mlx-lm PR #1788's
model file (text-only, parity-tested) as the base of `mlx_lm/models/qwen4_exp.py`
in our fork, cross-check every mechanism against Blaizzy/mlx-vlm's merged
implementation (the only MLX code with MTP + external PLE + QSA batching),
and use transformers v5.17.0 / vLLM #53896 as semantic references.

- **M0 — text-only dense path.** PR #1788 file adapted to our tree; run
  `mlx-community/Qwen3.8-Flash-Next-4bit` (post-`ac83bb4`); acceptance =
  coherent generation + parity with BatchGenerator. Dense attention is
  *exact* ≤ 2051 tokens, so short-prompt tests are reference-true even
  before QSA lands. PLE table mmap'd from the safetensors shards.
- **M1 — QSA eager indexer.** Compressed indexer-K cache, block pooling
  pre-RoPE, argpartition top-512, gather-based sparse SDPA; parity vs
  dense for ≤ 2051-token prompts (they must match exactly).
- **M2 — PLE engineering.** Quantize (4–6 bit) and/or external-file the
  table (ddalcu's 32 GB `ngram_table.bin` pattern / jjang-ai SSD-backed
  PLE), selective row dequant on lookup.
- **M3 — MTP.** Extract `mtp.*` into a loadable draft model wired as
  mlx-lm external `draft_model` first (no generate-loop surgery), then
  evaluate native same-checkpoint drafting with QSA index reuse.
- **M4 — Metal kernels + long context.** Fused HC/PLE/QSA Metal kernels
  only after the eager paths are correct; YaRN override for 1M context.

Open decisions for the user: (a) acceptance checkpoint choice, (b) whether
M3 targets external-draft-first (cheap) or native MTP, (c) PLE strategy on
this 128 GB M4 Max (in-RAM quantized ~26–32 GB vs external SSD file),
(d) whether to track upstream PR #1788 and rebase onto it if it merges.

## Source index

Worker outputs: `w1_transformers.txt` (transformers mechanics), `w2_vllm.txt`
(vLLM + kernel inventory), `w3_llamacpp.txt` (GGUF + traps),
`w4_mlx_ecosystem.txt` (MLX landscape + local baseline + core ops),
`w5_qwen_upstream.txt` (official/Megatron/sglang mechanics). Official
snapshots: `official_config.json`, `official_readme.md`. Primary URLs cited
inline above and in each worker output.
