import unittest

import mlx.core as mx

from mlx_lm.reasoning_entropy import (
    EntropyGate,
    EntropyGateConfig,
    GateState,
    make_entropy_gate,
)


def peaked_logits(vocab=1000):
    x = mx.full((vocab,), -100.0)
    x[0] = 10.0
    return x


def flat_logits(vocab=1000):
    return mx.zeros(vocab)


def moderate_logits(vocab=1000):
    x = mx.zeros(vocab)
    x[0]=5.0; x[1]=4.0; x[2]=3.0
    return x


class TestEntropyGate(unittest.TestCase):
    def test_peaked_distribution_has_low_entropy(self):
        gate = make_entropy_gate()
        h = gate.normalized_entropy(peaked_logits())
        self.assertLess(h, 0.01)

    def test_flat_distribution_has_high_entropy(self):
        gate = make_entropy_gate()
        h = gate.normalized_entropy(flat_logits())
        self.assertGreater(h, 0.99)

    def test_moderate_distribution_is_between(self):
        gate = make_entropy_gate()
        h = gate.normalized_entropy(moderate_logits())
        self.assertGreater(h, 0.01)
        self.assertLess(h, 0.99)

    def test_convergence_after_sustained_low_entropy(self):
        gate = make_entropy_gate(converge_window=3)
        for _ in range(10):
            state = gate.update(peaked_logits())
        self.assertEqual(state, GateState.CONVERGED)
        self.assertTrue(gate.should_terminate_reasoning())

    def test_no_convergence_with_mixed_entropy(self):
        gate = make_entropy_gate(converge_window=3)
        for i in range(10):
            logits = peaked_logits() if i % 2 == 0 else flat_logits()
            state = gate.update(logits)
        self.assertNotEqual(state, GateState.CONVERGED)

    def test_wandering_after_sustained_high_entropy(self):
        gate = make_entropy_gate(wandering_window=5)
        for _ in range(10):
            state = gate.update(flat_logits())
        self.assertEqual(state, GateState.WANDERING)
        self.assertTrue(gate.should_redirect())

    def test_reset_clears_state(self):
        gate = make_entropy_gate()
        for _ in range(10):
            gate.update(peaked_logits())
        gate.reset()
        self.assertEqual(gate.state.state, GateState.REASONING)
        self.assertEqual(gate.state.tokens_in_reasoning, 0)

    def test_disabled_gate_always_returns_reasoning(self):
        config = EntropyGateConfig(enabled=False)
        gate = EntropyGate(config)
        for _ in range(20):
            state = gate.update(flat_logits())
        self.assertEqual(state, GateState.REASONING)

    def test_stats_reports_correct_fields(self):
        gate = make_entropy_gate()
        gate.update(moderate_logits())
        s = gate.stats()
        self.assertIn("state", s)
        self.assertIn("tokens", s)
        self.assertIn("mean_entropy", s)
        self.assertIn("last_entropy", s)


if __name__ == "__main__":
    unittest.main()
