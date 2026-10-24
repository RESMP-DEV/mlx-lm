"""Shannon entropy gating for reasoning model output.

Measures next-token distribution entropy during reasoning and applies a
continue, converge, or redirect policy to reduce wasted thinking tokens.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mlx.core as mx


class GateState(Enum):
    REASONING = "reasoning"
    CONVERGED = "converged"
    WANDERING = "wandering"


@dataclass
class EntropyGateConfig:
    converge_threshold: float = 0.15
    converge_window: int = 3
    wandering_threshold: float = 0.75
    wandering_window: int = 20
    top_k: int = 64
    enabled: bool = False


@dataclass
class EntropyGateState:
    state: GateState = GateState.REASONING
    low_entropy_streak: int = 0
    high_entropy_streak: int = 0
    tokens_in_reasoning: int = 0
    entropy_history: list[float] = field(default_factory=list)


class EntropyGate:
    """Tracks entropy and emits gating decisions during reasoning."""

    def __init__(self, config: EntropyGateConfig):
        self.config = config
        self.state = EntropyGateState()

    def normalized_entropy(self, logits: mx.array) -> float:
        """Compute normalized Shannon entropy in [0, 1]."""
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        probs = mx.exp(log_probs)
        k = min(self.config.top_k, logits.shape[-1])
        top_indices = mx.argpartition(probs, -k, axis=-1)[..., -k:]
        top_probs = mx.take_along_axis(probs, top_indices, axis=-1)
        top_probs = top_probs / mx.sum(top_probs, axis=-1, keepdims=True)
        top_log_probs = mx.log(top_probs + 1e-10)
        entropy = -mx.sum(top_probs * top_log_probs, axis=-1)
        max_entropy = math.log(k)
        return float(entropy.item()) / max_entropy

    def update(self, logits: mx.array) -> GateState:
        """Process one token's logits and return the current gate state."""
        if not self.config.enabled:
            return GateState.REASONING

        h = self.normalized_entropy(logits)
        self.state.tokens_in_reasoning += 1
        self.state.entropy_history.append(h)

        if h < self.config.converge_threshold:
            self.state.low_entropy_streak += 1
        else:
            self.state.low_entropy_streak = 0

        if h > self.config.wandering_threshold:
            self.state.high_entropy_streak += 1
        else:
            self.state.high_entropy_streak = 0

        if (
            self.state.low_entropy_streak >= self.config.converge_window
            and self.state.tokens_in_reasoning > 8
        ):
            self.state.state = GateState.CONVERGED
        elif self.state.high_entropy_streak >= self.config.wandering_window:
            self.state.state = GateState.WANDERING
        else:
            self.state.state = GateState.REASONING

        return self.state.state

    def should_terminate_reasoning(self) -> bool:
        return self.state.state == GateState.CONVERGED

    def should_redirect(self) -> bool:
        return self.state.state == GateState.WANDERING

    def reset(self):
        self.state = EntropyGateState()

    def stats(self) -> dict[str, Any]:
        h = self.state.entropy_history
        return {
            "state": self.state.state.value,
            "tokens": self.state.tokens_in_reasoning,
            "mean_entropy": sum(h) / len(h) if h else 0.0,
            "last_entropy": h[-1] if h else 0.0,
            "low_streak": self.state.low_entropy_streak,
            "high_streak": self.state.high_entropy_streak,
        }


def make_entropy_gate(
    converge_threshold: float = 0.15,
    converge_window: int = 3,
    wandering_threshold: float = 0.75,
    wandering_window: int = 20,
) -> EntropyGate:
    config = EntropyGateConfig(
        converge_threshold=converge_threshold,
        converge_window=converge_window,
        wandering_threshold=wandering_threshold,
        wandering_window=wandering_window,
        enabled=True,
    )
    return EntropyGate(config)
