# Copyright © 2026 Apple Inc.

import os
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.qwen4_exp_ple import ExternalPleTable


class TestExternalPle(unittest.TestCase):
    SOURCE = (
        Path(os.environ["MLX_LM_QWEN4_EXP_PLE_FIXTURE"])
        if "MLX_LM_QWEN4_EXP_PLE_FIXTURE" in os.environ
        else None
    )

    def expected_rows(self, path, table, rows):
        weights = mx.load(str(path / "model-00001.safetensors"))
        expected = []
        for row in rows.reshape(-1):
            shard = row // table.rows_per_shard
            key = (
                "model.layers.1.ple.ple_embedding.ngram_embedding."
                f"shards.{shard}.weight"
            )
            expected.append(weights[key][row % table.rows_per_shard])
        return mx.stack(expected).reshape(*rows.shape, -1)

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_selected_rows_match_resident_shards(self):
        path = self.SOURCE
        table = ExternalPleTable(path, n_shards=4)
        self.addCleanup(table.close)
        rows = np.array([[0, 129, 258, 387, 511], [1, 130, 2, 3, 4]])

        actual = table.gather(rows)
        expected = self.expected_rows(path, table, rows)

        self.assertEqual(actual.dtype, mx.bfloat16)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(bool(mx.all(actual == expected).item()))

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_mmap_rows_match_pread_rows(self):
        path = self.SOURCE
        pread = ExternalPleTable(path, n_shards=4)
        mapped = ExternalPleTable(path, n_shards=4, read_backend="mmap")
        self.addCleanup(pread.close)
        self.addCleanup(mapped.close)
        rows = np.array([[0, 129, 258, 387, 511], [1, 130, 2, 3, 4]])

        actual = mapped.gather(rows)
        expected = pread.gather(rows)
        mx.eval(actual, expected)
        self.assertEqual(mapped.read_backend, "mmap")
        self.assertTrue(bool(mx.all(actual == expected).item()))

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_unknown_read_backend_is_rejected(self):
        path = self.SOURCE
        with self.assertRaisesRegex(ValueError, "unknown PLE read backend"):
            ExternalPleTable(path, n_shards=4, read_backend="invalid")

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_duplicate_rows_and_global_cache_keys(self):
        path = self.SOURCE
        table = ExternalPleTable(path, n_shards=4, cache_rows=8)
        self.addCleanup(table.close)
        rows = np.array([[0, 0, 1, 128, 128, 129, 257, 2, 2]])

        actual = table.gather(rows)
        expected = self.expected_rows(path, table, rows)
        self.assertTrue(bool(mx.all(actual == expected).item()))
        self.assertEqual(table.stats()["cache_size"], 6)
        for row in np.unique(rows):
            self.assertIsNone(table._cache_get(int(row)).base)
        first_stats = table.stats()
        self.assertEqual(first_stats["requested_rows"], 9)
        self.assertEqual(first_stats["disk_rows"], 6)
        self.assertEqual(first_stats["read_calls"], 3)
        self.assertEqual(first_stats["read_bytes"], 192)

        reads = 0
        original = table._read_rows

        def counted(*args, **kwargs):
            nonlocal reads
            reads += 1
            return original(*args, **kwargs)

        table._read_rows = counted
        actual = table.gather(rows)
        self.assertEqual(reads, 0)
        second_stats = table.stats()
        self.assertEqual(second_stats["cache_hits"], first_stats["cache_hits"] + 9)
        self.assertEqual(second_stats["disk_rows"], first_stats["disk_rows"])
        self.assertEqual(second_stats["read_bytes"], first_stats["read_bytes"])
        self.assertTrue(bool(mx.all(actual == expected).item()))

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_cache_capacity_can_shrink_and_grow(self):
        path = self.SOURCE
        table = ExternalPleTable(path, n_shards=4, cache_rows=4)
        self.addCleanup(table.close)
        table.gather(np.array([[0, 1, 2, 3]]))
        self.assertEqual(table.stats()["cache_capacity"], 4)
        self.assertEqual(table.stats()["cache_size"], 4)

        table.set_cache_capacity(2)
        stats = table.stats()
        self.assertEqual(stats["cache_capacity"], 2)
        self.assertEqual(stats["cache_size"], 2)

        table.set_cache_capacity(4)
        stats = table.stats()
        self.assertEqual(stats["cache_capacity"], 4)
        self.assertEqual(stats["cache_size"], 2)

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_issue_collect_matches_gather(self):
        path = self.SOURCE
        table = ExternalPleTable(path, n_shards=4, cache_rows=8)
        self.addCleanup(table.close)
        rows = np.array([[3, 130, 2, 258], [7, 7, 129, 0]])

        ticket = table.issue(rows)
        collected = table.collect(ticket)
        table.close()
        expected = self.expected_rows(path, table, rows)
        self.assertEqual(collected.shape, expected.shape)
        self.assertTrue(bool(mx.all(collected == expected).item()))
        self.assertEqual(table.stats()["gathers"], 1)

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_ticket_reports_shape_and_issue_timing(self):
        path = self.SOURCE
        table = ExternalPleTable(path, n_shards=4)
        self.addCleanup(table.close)
        rows = np.array([[1, 2, 3], [4, 5, 6]])
        ticket = table.issue(rows)
        self.assertEqual(ticket.shape, (2, 3))
        self.assertEqual(ticket.requested, 6)
        self.assertGreaterEqual(ticket.issue_ns, 0)
        table.collect(ticket)
        stats = table.stats()
        self.assertEqual(stats["gathers"], 1)
        self.assertEqual(stats["requested_rows"], 6)
        self.assertGreaterEqual(stats["elapsed_ns"], 0)
        self.assertGreaterEqual(stats["issue_ns"], 0)

    @unittest.skipUnless(SOURCE and SOURCE.exists(), "PLE fixture is unavailable")
    def test_empty_request_is_a_noop(self):
        path = self.SOURCE
        table = ExternalPleTable(path, n_shards=4)
        self.addCleanup(table.close)
        ticket = table.issue(np.zeros((0, 16), dtype=np.int64))
        values = table.collect(ticket)
        mx.eval(values)
        self.assertEqual(values.shape, (0, 16, table.values_per_row))
        self.assertEqual(table.stats()["requested_rows"], 0)


class TestSharedExpertFusion(unittest.TestCase):
    def test_mxfp4_gate_up_fusion_is_exact(self):
        from mlx_lm.models.qwen4_exp import MLP

        mlp = MLP(64, 64)
        nn.quantize(mlp, group_size=32, bits=4, mode="mxfp4")
        x = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
        before = mlp(x)
        self.assertTrue(mlp.fuse_quantized())
        after = mlp(x)
        mx.eval(before, after)
        self.assertTrue(bool(mx.all(before == after).item()))
        self.assertIsNone(mlp.up_proj)


class TestAttentionKVFusion(unittest.TestCase):
    def test_mxfp4_kv_fusion_is_exact(self):
        from mlx_lm.models.qwen4_exp import Attention, TextArgs

        args = TextArgs(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-6,
            indexer_n_heads=2,
            indexer_head_dim=16,
            indexer_kv_heads=1,
            indexer_budget=8,
            indexer_compress_ratio=4,
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            rope_parameters={},
        )
        attention = Attention(args)
        nn.quantize(attention, group_size=32, bits=4, mode="mxfp4")
        x = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
        k, v = attention.k_proj(x), attention.v_proj(x)
        self.assertTrue(attention.fuse_quantized())
        fused = attention.k_proj(x)
        kf, vf = mx.split(fused, 2, axis=-1)
        mx.eval(k, v, kf, vf)
        self.assertTrue(bool(mx.all(k == kf).item()))
        self.assertTrue(bool(mx.all(v == vf).item()))
        self.assertIsNone(attention.v_proj)

    def test_compiled_output_matches_eager(self):
        from mlx_lm.models.qwen4_exp import Attention, TextArgs

        args = TextArgs(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-6,
            indexer_n_heads=2,
            indexer_head_dim=16,
            indexer_kv_heads=1,
            indexer_budget=8,
            indexer_compress_ratio=4,
            rope_theta=10000.0,
            partial_rotary_factor=0.25,
            rope_parameters={},
        )
        attention = Attention(args)
        nn.quantize(attention, group_size=32, bits=4, mode="mxfp4")
        attention.fuse_quantized()
        out = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
        gate = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
        eager = attention._output(out, gate)
        attention.compile_decode()
        compiled = attention.compiled_output(out, gate)
        mx.eval(eager, compiled)
        self.assertTrue(bool(mx.all(eager == compiled).item()))


class TestGDNProjectionFusion(unittest.TestCase):
    def test_decode_conv_cache_kernel_is_bitwise(self):
        from mlx_lm.models.qwen4_exp import _gdn_conv_decode_update

        mx.random.seed(2509)
        channels, width = 10240, 4
        state = mx.random.normal((1, width - 1, channels), dtype=mx.bfloat16)
        current = mx.random.normal((1, 1, channels), dtype=mx.bfloat16)
        conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=width,
            groups=channels,
            bias=False,
        )
        conv.weight = mx.random.normal((channels, width, 1), dtype=mx.bfloat16) * 0.2
        full_input = mx.concatenate([state, current], axis=1)
        expected = nn.silu(conv(full_input))
        expected_state = mx.contiguous(full_input[:, -(width - 1) :, :])
        actual, actual_state = _gdn_conv_decode_update(state, current, conv.weight)
        mx.eval(expected, expected_state, actual, actual_state)

        self.assertTrue(bool(mx.all(expected == actual).item()))
        self.assertTrue(bool(mx.all(expected_state == actual_state).item()))

    def test_mxfp4_projection_fusion_is_exact(self):
        from mlx_lm.models.qwen4_exp import GatedDeltaNet, TextArgs

        args = TextArgs(
            hidden_size=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        gdn = GatedDeltaNet(args)
        nn.quantize(gdn, group_size=32, bits=4, mode="mxfp4")
        x = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
        qkv, z = gdn.in_proj_qkv(x), gdn.in_proj_z(x)
        b, a = gdn.in_proj_b(x), gdn.in_proj_a(x)
        self.assertEqual(gdn.fuse_quantized(), 2)
        qkvz = gdn.in_proj_qkv(x)
        qkvf, zf = mx.split(qkvz, [gdn.conv_dim], axis=-1)
        baf = gdn.in_proj_b(x)
        bf, af = mx.split(baf, 2, axis=-1)
        mx.eval(qkv, z, b, a, qkvf, zf, bf, af)
        self.assertTrue(bool(mx.all(qkv == qkvf).item()))
        self.assertTrue(bool(mx.all(z == zf).item()))
        self.assertTrue(bool(mx.all(b == bf).item()))
        self.assertTrue(bool(mx.all(a == af).item()))
        self.assertIsNone(gdn.in_proj_z)
        self.assertIsNone(gdn.in_proj_a)

    def test_all_projection_fusion_is_exact(self):
        from mlx_lm.models.qwen4_exp import GatedDeltaNet, TextArgs

        args = TextArgs(
            hidden_size=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        gdn = GatedDeltaNet(args)
        nn.quantize(gdn, group_size=32, bits=4, mode="mxfp4")
        self.assertEqual(gdn.fuse_quantized(), 2)
        x = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
        mask = mx.ones((1, 1), dtype=mx.bool_)
        before = gdn(x, mask, None)

        gdn.fuse_all_projections = True
        self.assertEqual(gdn.fuse_quantized(), 1)
        after = gdn(x, mask, None)
        mx.eval(before, after)
        self.assertTrue(bool(mx.all(before == after).item()))
        self.assertTrue(gdn.all_fused)
        self.assertIsNone(gdn.in_proj_b)

    def test_prefused_gdn_matches_runtime_fusion(self):
        from mlx_lm.models.qwen4_exp import GatedDeltaNet, TextArgs

        source_args = TextArgs(
            hidden_size=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            fuse_gdn_all=True,
        )
        target_args = TextArgs(
            hidden_size=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            prefused_gdn=True,
        )
        source = GatedDeltaNet(source_args)
        target = GatedDeltaNet(target_args)
        nn.quantize(source, group_size=32, bits=4, mode="mxfp4")
        nn.quantize(target, group_size=32, bits=4, mode="mxfp4")

        self.assertEqual(source.fuse_quantized(), 3)
        self.assertEqual(target.fuse_quantized(), 0)
        target.update(source.parameters())
        x = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
        mask = mx.ones((1, 1), dtype=mx.bool_)
        expected = source(x, mask, None)
        actual = target(x, mask, None)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.all(expected == actual).item()))
        self.assertTrue(target.all_fused)
        self.assertIsNone(target.in_proj_z)
        self.assertIsNone(target.in_proj_b)
        self.assertIsNone(target.in_proj_a)

    def test_compiled_output_matches_eager(self):
        from mlx_lm.models.qwen4_exp import GatedDeltaNet, TextArgs

        args = TextArgs(
            hidden_size=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        gdn = GatedDeltaNet(args)
        nn.quantize(gdn, group_size=32, bits=4, mode="mxfp4")
        out = mx.random.normal((1, 1, 4, 16), dtype=mx.bfloat16)
        z = mx.random.normal((1, 1, 4, 16), dtype=mx.bfloat16)
        eager = gdn._output(out, z)
        gdn.compile_decode()
        compiled = gdn._compiled_output(out, z)
        mx.eval(eager, compiled)
        self.assertTrue(bool(mx.all(eager == compiled).item()))


class TestCompiledGatedResidual(unittest.TestCase):
    def _args(self):
        from mlx_lm.models.qwen4_exp import TextArgs

        return TextArgs(hidden_size=32, hc_count=4, hc_lowrank=8)

    def test_compiled_decode_matches_eager(self):
        from mlx_lm.models.qwen4_exp import GatedResidual

        residual = GatedResidual(self._args())
        hyper = mx.random.normal((1, 1, 128), dtype=mx.bfloat16)
        eager_mixed, eager_hyper, eager_inject = residual._forward(hyper)
        residual.compile_decode()
        mixed, kept_hyper, inject = residual(hyper)
        mx.eval(eager_mixed, eager_hyper, eager_inject, mixed, kept_hyper, inject)

        self.assertTrue(bool(mx.all(eager_mixed == mixed).item()))
        self.assertTrue(bool(mx.all(eager_hyper == kept_hyper).item()))
        self.assertTrue(bool(mx.all(eager_inject == inject).item()))

    def test_batched_decode_stays_eager(self):
        from mlx_lm.models.qwen4_exp import GatedResidual

        residual = GatedResidual(self._args(), use_combine=False)
        residual.compiled_forward = lambda _x: self.fail("batched decode used compile")
        batched = mx.zeros((2, 1, 128), dtype=mx.bfloat16)
        mixed = residual(batched)
        expected = residual._forward(batched)
        mx.eval(mixed, expected)
        self.assertTrue(bool(mx.all(mixed == expected).item()))


class TestCompiledInjection(unittest.TestCase):
    def test_compiled_injection_matches_eager(self):
        from mlx_lm.models.qwen4_exp import _compiled_inject_block, _inject_block

        x = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)
        hyper = mx.random.normal((1, 1, 256), dtype=mx.bfloat16)
        inject = mx.random.normal((1, 1, 4), dtype=mx.bfloat16)
        eager = _inject_block(x, hyper, inject)
        compiled = _compiled_inject_block(x, hyper, inject)
        mx.eval(eager, compiled)
        self.assertTrue(bool(mx.all(eager == compiled).item()))


class TestPrefusedSwitchGLU(unittest.TestCase):
    def test_prefused_gate_up_gather_is_exact(self):
        from mlx_lm.models.switch_layers import SwitchGLU

        source = SwitchGLU(64, 32, 3)
        nn.quantize(source, group_size=32, bits=4, mode="mxfp4")
        target = SwitchGLU(64, 32, 3, fused_gate_up=True)
        nn.quantize(target, group_size=32, bits=4, mode="mxfp4")
        target.gate_proj.weight = mx.concatenate(
            [source.gate_proj.weight, source.up_proj.weight], axis=1
        )
        target.gate_proj.scales = mx.concatenate(
            [source.gate_proj.scales, source.up_proj.scales], axis=1
        )
        target.down_proj.weight = source.down_proj.weight
        target.down_proj.scales = source.down_proj.scales

        x = mx.random.normal((4, 64), dtype=mx.bfloat16)
        indices = mx.array([[0, 1], [1, 2], [2, 0], [1, 1]])
        expected = source(x, indices)
        actual = target(x, indices)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.all(expected == actual).item()))
        self.assertIsNone(target.up_proj)


class TestNGramHashBackends(unittest.TestCase):
    """Host and MLX n-gram hashing must agree exactly."""

    def _embedding(self):
        from mlx_lm.models.qwen4_exp import NGramEmbedding, TextArgs

        args = TextArgs(
            ngram_size=3,
            heads_per_ngram=8,
            ngram_vocab_size_base=20000000,
            split_ngram_parts=4,
            make_ngram_vocab_size_divisible_by=128,
            eos_token_id=1,
            seed=17,
            ple_embed_dim=32,
            vocab_size=1000,
        )
        return NGramEmbedding(args, 16, 0)

    def test_host_and_mlx_hashes_match(self):
        embedding = self._embedding()
        ids = mx.array([[5, 9, 13, 21, 34, 55, 89, 144]])
        previous = mx.array([[1, 1]])
        host = embedding._host_gids(ids, previous)
        mlx = embedding._mlx_gids(ids, previous)
        mx.eval(mlx)
        self.assertTrue(bool(mx.all(mx.array(host) == mlx).item()))
        self.assertEqual(host.shape, (1, 8, 16))

        embedding.hash_backend = "auto"
        single = embedding._gid(ids, previous)
        batch_ids = mx.concatenate([ids, ids], axis=0)
        batch_previous = mx.concatenate([previous, previous], axis=0)
        batch = embedding._gid(batch_ids, batch_previous)
        mx.eval(single, batch)
        self.assertTrue(bool(mx.all(mx.array(host) == single).item()))
        self.assertEqual(batch.shape, (2, 8, 16))


class TestBatchedPromptCacheReset(unittest.TestCase):
    def test_batch_attention_cache_reset_clears_state(self):
        from mlx_lm.models.qwen4_exp import _BatchAttnCache

        cache = _BatchAttnCache([1, 2])
        cache.reset()

        self.assertTrue(cache.empty())
        self.assertEqual(cache.size(), 0)
        self.assertIsNone(cache.indexer.keys)
        self.assertTrue(bool(mx.all(cache.offset == 0).item()))


class TestHybridBatchExtend(unittest.TestCase):
    def test_extend_unequal_batches_preserves_rows(self):
        from mlx_lm.models.qwen4_exp import _BatchAttnCache

        wide = _BatchAttnCache([0, 0])
        narrow = _BatchAttnCache([0])
        keys = mx.random.normal((2, 2, 1, 16), dtype=mx.bfloat16)
        values = mx.random.normal((2, 2, 1, 16), dtype=mx.bfloat16)
        wide.update_and_fetch(keys, values)
        wide.indexer.update(mx.random.normal((2, 1, 8), dtype=mx.bfloat16))
        narrow.update_and_fetch(keys[:1], values[:1])
        narrow.indexer.update(mx.random.normal((1, 1, 8), dtype=mx.bfloat16))

        wide.extend(narrow)

        self.assertEqual(wide.offset.shape, (3,))
        self.assertEqual(wide.keys.shape[0], 3)
        self.assertEqual(wide.values.shape[0], 3)
        self.assertEqual(wide.indexer.keys.shape[0], 3)


class TestHybridCacheSerialization(unittest.TestCase):
    def test_qwen_cache_classes_round_trip(self):
        import tempfile

        from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
        from mlx_lm.models.qwen4_exp import _AttnCache, _LayerCache

        cache = [_AttnCache(), _LayerCache(4)]
        keys = mx.random.normal((1, 2, 1, 16), dtype=mx.bfloat16)
        values = mx.random.normal((1, 2, 1, 16), dtype=mx.bfloat16)
        cache[0].update_and_fetch(keys, values)
        cache[0].indexer.update(mx.random.normal((1, 1, 8), dtype=mx.bfloat16))
        cache[1][0] = mx.random.normal((1, 1, 4), dtype=mx.bfloat16)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen-cache.safetensors"
            save_prompt_cache(path, cache)
            loaded = load_prompt_cache(path)

        self.assertEqual([type(item) for item in loaded], [_AttnCache, _LayerCache])
        mx.eval(loaded[0].keys, cache[0].keys)
        self.assertTrue(bool(mx.array_equal(loaded[0].keys, cache[0].keys).item()))
        self.assertTrue(
            bool(mx.array_equal(loaded[0].indexer.keys, cache[0].indexer.keys).item())
        )
        self.assertTrue(bool(mx.array_equal(loaded[1][0], cache[1][0]).item()))


if __name__ == "__main__":
    unittest.main()
