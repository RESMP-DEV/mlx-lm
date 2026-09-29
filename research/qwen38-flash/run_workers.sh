#!/usr/bin/env bash
# Serialized Qwen3.8-Flash-Next research workers: run one at a time against
# the shared GLM-5.3-FlashX provider cooldown. w5 is already running elsewhere.
set -u
BASE=/Users/kearm/mlx-lm/research/qwen38-flash
LOG="$BASE/run_workers.log"

log() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$1" >> "$LOG"; }

START=$(date +%s)
log "orchestrator start; cooldown margin before serialized queue"
sleep 120
log "starting serialized workers"

run_w() {
  name="$1"; out="$2"; brief="$3"
  attempt=1
  while [ "$attempt" -le 3 ]; do
    log "$name attempt $attempt"
    codex-ccr-worker --cwd /Users/kearm/mlx-lm --read-only --allow-network --timeout 1500 --output "$out" "$brief" >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then log "$name ok"; return 0; fi
    log "$name failed rc=$rc"
    if [ "$rc" -ne 75 ]; then return "$rc"; fi
    sleep 240
    attempt=$((attempt + 1))
  done
  return 1
}

W1='Parent context: we forked mlx and mlx-lm (checkouts /Users/kearm/mlx and /Users/kearm/mlx-lm) to port Qwen/Qwen3.8-Flash-Next. NVIDIA Megatron calls it qwen4_exp. Baseline lineage in mlx-lm is qwen3_next. Research only, read-only; gh CLI is authenticated; network use is authorized.

Known already (parent fetched the official config.json, do not restate as discovery): model_type qwen4_exp (VLM wrapper Qwen4ExpForConditionalGeneration, text_config model_type qwen4_exp_text, transformers_version 5.8.0.dev0); 48 layers with full_attention_interval 4; linear attention 16 key heads / 48 value heads dims 128, conv kernel 4; full attention head_dim 256, 24 q heads, 2 kv heads, partial_rotary_factor 0.25, rope_theta 1e7, mrope_interleaved; MoE 512 experts top-10, moe_intermediate 640, shared expert 640; ngram_size 3, heads_per_ngram 8, ngram_vocab_size_base 20000000, split_ngram_parts 128, make_ngram_vocab_size_divisible_by 128; indexer_budget 2048, indexer_head_dim 128, indexer_n_heads 4, indexer_kv_heads 1, indexer_compress_ratio 4; hc_count 4, hc_lowrank 320; ple_layer_ids [2], ple_embed_dim 2560, ple_conv_kernel_size 4; MTP 1 full-attention layer, hybrid; vocab 248320; max_position 262144. Model card states 125B total, 6B active, plus 51B n-gram and 4B MTP parameters.

Your job: explain HOW the new mechanisms work in code, in huggingface/transformers.

Questions:
1. Locate the implementation: gh search prs --repo huggingface/transformers --state all with queries qwen4_exp, qwen3.8, "flash next", flash-next, qwen3_8; also gh api to list src/transformers/models directories matching qwen4. Fetch modeling_qwen4_exp*.py and configuration_qwen4_exp*.py from the matching branch (curl raw.githubusercontent.com). Which release or dev revision carries it.
2. N-gram head mechanics: how ngram_size, heads_per_ngram, ngram_vocab_size_base, and split_ngram_parts combine. Is the 20M ngram vocab an embedding table over token n-grams; how are n-gram ids computed (hash or product form); what does split_ngram_parts do at runtime; what tensors does this add to the checkpoint and how large are they; is the head used for inference-time speculation or for a training loss only.
3. Indexer mechanics: what the indexer attends over, how indexer_budget 2048 and compress_ratio 4 produce a token-selection mask feeding the full-attention layers, and the exact ops used.
4. Hyper connections: what hc_count 4 and hc_lowrank 320 do to residual stream math (static or learned mixing, where the lowrank projections live).
5. PLE module: what the ple_layer_ids [2] module computes and how it feeds routing or attention.
6. Precise delta list vs the transformers qwen3_next model directory, function by function.

Output contract: markdown to your output file. Sections: Summary (max 6 bullets); Findings (numbered, each with URL or PR number, substance, decisive code excerpts); New mechanism mechanics (ngram head, indexer, hc, PLE, MTP: one subsection each with the math or pseudocode); Architecture delta vs qwen3_next (table); MLX port implications; Open questions. Cite everything. Only write your output file; do not modify repository files.'

W2='Parent context: we forked mlx and mlx-lm (checkouts /Users/kearm/mlx and /Users/kearm/mlx-lm) to port Qwen/Qwen3.8-Flash-Next. NVIDIA Megatron calls it qwen4_exp. Baseline lineage in mlx-lm is qwen3_next. Research only, read-only; gh CLI is authenticated; network use is authorized.

Known already (parent fetched official config.json, do not restate): qwen4_exp VLM wrapper, text qwen4_exp_text; 48 layers full_attention_interval 4; linear attn 16k/48v dim 128 conv 4; full attn head_dim 256, 24 q, 2 kv, partial rotary 0.25; MoE 512 experts top-10; ngram_size 3, heads_per_ngram 8, ngram_vocab_size_base 20000000, split_ngram_parts 128; indexer_budget 2048; hc_count 4 hc_lowrank 320; ple_layer_ids [2]; MTP 1 layer; vocab 248320.

Scope: vLLM main repo plus vLLM-adjacent kernel stacks.

Questions:
1. vllm-project/vllm support status: gh search prs --repo vllm-project/vllm --state all with queries qwen3.8, "flash next", flash-next, qwen4_exp, qwen3_8; also gh search issues. Identify the model file under vllm/model_executor/models, registry mapping, merged vs pending; fetch with gh pr diff.
2. New kernels or ops vs plain qwen3_next in vLLM: chunked linear attention, delta rule, indexer top-k selection, n-gram head ops, MTP speculative support, fp8 paths; check PR diffs and new files under vllm/attention or csrc.
3. Corroborating stacks: vllm-project/vllm-ascend PR 16981, sgl-project/sgl-kernel-npu PR 807 (Triton kernels), flagos-ai/vllm-plugin-FL PR 455 (H100), ROCm/aiter PR 5820 (FlyDSL gfx950). Use gh pr view and gh pr diff. List every op implemented for this model: op name, purpose, source PR.
4. Serving details: how the n-gram head and MTP are used at decode time (speculative? lossless n-gram lookahead?), linear attention chunk size and state dtype, KV layout changes vs qwen3_next, indexer top-k flow.

Output contract: markdown to your output file. Sections: Summary (max 6 bullets); Findings (numbered, PR number or URL, substance, decisive diff excerpts); New ops and kernels required (table: op name, purpose, source PR); Architecture delta vs qwen3_next; MLX port implications (which ops already exist in MLX core at /Users/kearm/mlx/mlx, which need Metal kernels); Open questions. Cite everything. Only write your output file; do not modify repository files.'

W3='Parent context: we forked mlx and mlx-lm (checkouts /Users/kearm/mlx and /Users/kearm/mlx-lm) to port Qwen/Qwen3.8-Flash-Next. NVIDIA Megatron calls it qwen4_exp. Baseline lineage in mlx-lm is qwen3_next. Research only, read-only; gh CLI is authenticated; network use is authorized.

Known already (parent fetched official config.json, do not restate): qwen4_exp VLM wrapper, text qwen4_exp_text; 48 layers full_attention_interval 4; linear attn 16k/48v dim 128 conv 4; full attn head_dim 256, 24 q, 2 kv, partial rotary 0.25; MoE 512 experts top-10; ngram_size 3, heads_per_ngram 8, ngram_vocab_size_base 20000000, split_ngram_parts 128; indexer_budget 2048; hc_count 4; ple_layer_ids [2]; MTP 1 layer; vocab 248320.

Scope: llama.cpp and GGUF plus two Rust ports for cross-checking.

Questions:
1. ggml-org/llama.cpp PR 28243 models: Qwen3.8-Flash-Next MTP: gh pr view and gh pr diff. Also gh search prs --repo ggml-org/llama.cpp --state all with queries qwen3.8, qwen4_exp, "flash next". Report the new GGUF arch id string, tensor naming delta vs qwen3next, convert_hf_to_gguf.py changes: tensors added, renamed, skipped; how the n-gram head tensors (20M vocab embedding) and MTP weights are handled.
2. KV cache and state handling changes: recurrent SSM state size for linear layers, new KV or cache flags, rope or mask changes vs qwen3next, indexer-related cache changes.
3. Cross-check antirez/ds4 PR 1115 and EricLBuehler/mistral.rs PR 2420 via gh pr view and gh pr diff: bugs and subtleties each hit (numeric issues, layout mismatches, tokenizer quirks, MTP handling, n-gram head skipping). These are traps an MLX port must avoid.
4. unsloth/Qwen3.8-Flash-Next-GGUF and Qwen/Qwen3.8-Flash-Next via curl https://huggingface.co/api/models/<id>: tensor inventory, arch metadata, total size on disk, and whether n-gram or MTP tensors ship in GGUFs at all.

Output contract: markdown to your output file. Sections: Summary (max 6 bullets); Findings (numbered, PR or URL, substance, decisive diff excerpts); GGUF tensor naming delta vs qwen3next (table); Bugs and traps found in other ports; MLX port implications; Open questions. Cite everything. Only write your output file; do not modify repository files.'

W4='Parent context: we forked mlx and mlx-lm (checkouts /Users/kearm/mlx and /Users/kearm/mlx-lm) to port Qwen/Qwen3.8-Flash-Next. NVIDIA Megatron calls it qwen4_exp. Baseline lineage in mlx-lm is qwen3_next. Research only, read-only; gh CLI is authenticated; network use is authorized. The official config.json is saved at /Users/kearm/mlx-lm/research/qwen38-flash/official_config.json for reference; do not restate its values as findings.

Scope: the MLX ecosystem, existing MLX conversions, and the local porting baseline.

Questions:
1. Existing MLX conversions on HF. For ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit, tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ, Jundot/Qwen3.8-Flash-Next-oQ4e-mtp: fetch curl https://huggingface.co/api/models/<id> (siblings list included) plus /resolve/main/config.json and README.md. Report declared model_type or arch string, which runtime runs them (mlx-lm fork, mlx-serve, custom code, linked GitHub repo), quant formats, custom code files shipped (files ending .py), and whether n-gram or MTP or indexer weights are present in the file list.
2. Search for MLX support work: gh search prs --repo ml-explore/mlx-lm --state all with queries qwen3.8, flash, qwen4, qwen3_8; same for ml-explore/mlx; and gh search prs across GitHub for "qwen3.8 mlx" and "qwen4_exp mlx". Check the mlx-community org for conversions. Report anything that shows a working mlx-lm port exists and where its code lives.
3. Local baseline study, read-only from /Users/kearm/mlx-lm: read mlx_lm/models/qwen3_next.py fully. Summarize: class names, gated delta net or linear attention implementation (which MLX ops it calls), layer composition from layer_types, cache structure for recurrent state, quantization hooks, MTP support if any. Then find the model registry (model_type to module mapping, likely mlx_lm/models/__init__.py or mlx_lm/models/cache.py or utils.py) and list exactly what a new qwen4_exp entry must touch: model file, registry, tokenizer wiring, convert script (mlx_lm/convert.py or similar), and any generate loop changes for MTP.
4. In /Users/kearm/mlx/mlx (core MLX, read-only): check for ops the port may need beyond what qwen3_next.py already uses: chunked cumsum, associative scan, top-k over keys, gather along sequence for indexer selection. rg for candidate ops; report what exists and what is missing.

Output contract: markdown to your output file. Sections: Summary (max 6 bullets); Findings (numbered, URL or file path, substance); Existing MLX conversions inventory (table: repo, arch string, runtime, quants, custom code, ngram or MTP or indexer weights present); qwen3_next baseline structure and registry wiring in mlx-lm (exact file paths and what a port touches); MLX core op availability; Open questions. Cite everything. Only write your output file; do not modify repository files.'

W5='Parent context: we forked mlx and mlx-lm (checkouts /Users/kearm/mlx and /Users/kearm/mlx-lm) to port Qwen/Qwen3.8-Flash-Next. NVIDIA Megatron calls it qwen4_exp. Baseline lineage in mlx-lm is qwen3_next. Research only, read-only; gh CLI is authenticated; network use is authorized.

Known already (parent fetched the official config.json, do not restate): model_type qwen4_exp (VLM wrapper, text qwen4_exp_text); 48 layers, full_attention_interval 4; linear attention 16k/48v heads dim 128; full attention head_dim 256, 24 q heads, 2 kv heads, partial rotary 0.25; MoE 512 experts top-10; ngram_size 3, heads_per_ngram 8, ngram_vocab_size_base 20000000; indexer_budget 2048; hc_count 4; ple_layer_ids [2]; MTP 1 layer; vocab 248320. Model card already states 125B total, 6B active, plus 51B n-gram and 4B MTP parameters: verify that quickly, then move past it.

Your job: authoritative architecture description from official and Megatron sources.

Questions:
1. Official model card: curl https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/README.md and the FP8 variant README. Architecture prose about the n-gram prediction mechanism, indexer, hyper connections, hybrid attention, context length, tokenizer, license, release date.
2. QwenLM GitHub: gh repo list QwenLM --limit 60; find the Qwen3.8 or Qwen4 repo; read release notes or docs mentioning Flash-Next.
3. NVIDIA/Megatron-LM PR 7393 feat: Qwen4-Exp (Qwen3.8-Flash-Next): gh pr view and gh pr diff. Report the Megatron implementation: layer pattern, attention classes, n-gram head implementation (how the 20M vocab embedding is handled in training), indexer implementation, hyper connection implementation, MTP, and the exact delta vs Megatron qwen3_next support.
4. NovaSky-AI/SkyRL PR 2216 megatron qwen3.8 flash next qwen4_exp: what they changed or fixed vs the Megatron PR.
5. sglang PR 36664 (Qwen3.8-Max residue inference) and 39724 (Qwen3.8 context parallel prefill): architecture facts about the family and Flash-Next they reveal, e.g. residue inference mechanism, MTP draft length, linear attention state size, how the n-gram head is used at inference.
6. State plainly from primary sources what is new or changed vs qwen3_next, and how the model is meant to be served fast: what role the n-gram head plays in decoding (lossless or speculative), and what the indexer replaces or supplements.

Output contract: markdown to your output file. Sections: Summary (max 6 bullets); Findings (numbered, each with URL or PR number, substance, decisive excerpts); New mechanism mechanics from official or Megatron sources (ngram head, indexer, hc, PLE, MTP); Family facts (params, layers, tokenizer, context, license, dates); MLX port implications; Open questions. Cite everything. Only write your output file; do not modify repository files.'

run_w w1 "$BASE/w1_transformers.txt" "$W1"
run_w w5 "$BASE/w5_qwen_upstream.txt" "$W5"
run_w w2 "$BASE/w2_vllm.txt" "$W2"
run_w w3 "$BASE/w3_llamacpp.txt" "$W3"
run_w w4 "$BASE/w4_mlx_ecosystem.txt" "$W4"
log "all done"
