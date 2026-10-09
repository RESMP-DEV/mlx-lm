# Copyright © 2026 Apple Inc.

import tempfile
import os
import unittest
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.generate import generate_step
from mlx_lm.quant.ple_runtime import build_external_runtime
from mlx_lm.sample_utils import greedy_sampler
from mlx_lm.utils import load_model


class TestPleRuntime(unittest.TestCase):
    SOURCE = (
        Path(os.environ["MLX_LM_QWEN4_EXP_PLE_FIXTURE"])
        if "MLX_LM_QWEN4_EXP_PLE_FIXTURE" in os.environ
        else None
    )

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_external_runtime_matches_resident_model(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            runtime = Path(temporary_dir) / "runtime"
            receipt = build_external_runtime(self.SOURCE, runtime, cache_rows=128)
            self.assertEqual(receipt["removed_ple_keys"], 4)
            self.assertEqual(receipt["pruned_shards"], ["model-00001.safetensors"])
            self.assertEqual(
                (runtime / "tokenizer.json").read_bytes(),
                (self.SOURCE / "tokenizer.json").read_bytes(),
            )

            resident, _ = load_model(self.SOURCE, lazy=True)
            external, _ = load_model(runtime, lazy=True)
            try:
                resident_keys = {k for k, _ in tree_flatten(resident.parameters())}
                external_keys = {k for k, _ in tree_flatten(external.parameters())}
                removed = {
                    k
                    for k in resident_keys - external_keys
                    if ".ngram_embedding.shards." in k
                }
                self.assertEqual(len(removed), 4)
                self.assertEqual(resident_keys - external_keys, removed)

                weights = mx.load(str(runtime / "model-00001.safetensors"))
                self.assertFalse(
                    any(".ngram_embedding.shards." in key for key in weights)
                )

                inputs = mx.array([[1, 2, 3, 4, 5, 6, 7]])
                resident_logits = resident(inputs)
                cache = external.make_cache()
                external_logits = external(inputs, cache=cache)
                mx.eval(resident_logits, external_logits)
                self.assertEqual(resident_logits.shape, external_logits.shape)
                self.assertTrue(bool(mx.all(resident_logits == external_logits).item()))

                ple = external.layers[1].ple.ple_embedding.ngram_embedding.external
                self.assertGreater(ple.stats()["cache_size"], 0)
                external.clear_ple_cache()
                self.assertEqual(ple.stats()["cache_size"], 0)
                external.reset_prompt_cache(cache)
                self.assertTrue(all(layer_cache.empty() for layer_cache in cache))

                reset_logits = external(inputs, cache=cache)
                mx.eval(reset_logits)
                self.assertTrue(bool(mx.all(resident_logits == reset_logits).item()))

                def greedy_ids(model):
                    prompt_cache = model.make_cache()
                    steps = generate_step(
                        inputs.reshape(-1),
                        model,
                        max_tokens=4,
                        sampler=greedy_sampler,
                        greedy_fast=True,
                        prompt_cache=prompt_cache,
                    )
                    ids = []
                    for token, _ in steps:
                        ids.append(token.item() if hasattr(token, "item") else token)
                    model.reset_prompt_cache(prompt_cache)
                    return ids

                self.assertEqual(greedy_ids(resident), greedy_ids(external))
            finally:
                external.close()


if __name__ == "__main__":
    unittest.main()
