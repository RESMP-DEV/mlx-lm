# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generate_step


class _NoCache:
    state = None

    def prepare(self, lengths=None, **kwargs):
        self.lengths = lengths

    def finalize(self):
        pass


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.prepared_prompt_tokens = None

    def prepare_prompt_tokens(self, total_tokens):
        self.prepared_prompt_tokens = total_tokens

    def make_cache(self):
        return [_NoCache()]

    def __call__(self, ids, cache=None, input_embeddings=None):
        self.calls += 1
        value = mx.sum(ids).astype(mx.float32)
        logits = mx.array([0.1, 2.0, -1.0, 0.3]) + value
        return mx.broadcast_to(logits, (1, ids.shape[1], 4))


class TestGreedyFast(unittest.TestCase):
    def tokens(self, greedy_fast):
        model = FixedLogits()
        generator = generate_step(
            mx.array([1, 2]),
            model,
            max_tokens=3,
            greedy_fast=greedy_fast,
        )
        return [(token, scores) for token, scores in generator]

    def test_fast_path_matches_default_tokens(self):
        default = self.tokens(False)
        fast = self.tokens(True)
        self.assertEqual([token for token, _ in default], [token for token, _ in fast])

        for (_, raw), (_, normalized) in zip(fast, default):
            expected = raw - mx.logsumexp(raw, keepdims=True)
            mx.eval(normalized, expected)
            self.assertTrue(bool(mx.allclose(normalized, expected).item()))

    def test_fast_path_rejects_other_sampler(self):
        with self.assertRaises(ValueError):
            list(
                generate_step(
                    mx.array([1]),
                    FixedLogits(),
                    max_tokens=1,
                    sampler=lambda scores: mx.argmax(scores, axis=-1),
                    greedy_fast=True,
                )
            )

    def test_prompt_preparation_receives_prompt_length(self):
        model = FixedLogits()
        list(generate_step(mx.array([1, 2]), model, max_tokens=1, greedy_fast=True))
        self.assertEqual(model.prepared_prompt_tokens, 2)


if __name__ == "__main__":
    unittest.main()
