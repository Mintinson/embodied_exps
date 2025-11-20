"""Exploration strategies for RL agents."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from rl_models.core.base import BaseExplorationStrategy


class EpsilonGreedyStrategy(BaseExplorationStrategy):
    """Epsilon-greedy exploration strategy with decay."""

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(
        self,
        state: torch.Tensor,
        action_selector: Callable[[torch.Tensor], int],
        env_action_space: Any,
    ) -> int:
        """Select action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return env_action_space.sample()
        return action_selector(state)

    def update(self) -> None:
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return self.epsilon


class GreedyStrategy(BaseExplorationStrategy):
    """Pure greedy strategy (no exploration)."""

    def select_action(
        self,
        state: torch.Tensor,
        action_selector: Callable[[torch.Tensor], int],
        env_action_space: Any,
    ) -> int:
        """Always select the greedy action."""
        return action_selector(state)

    def update(self) -> None:
        """No-op for greedy strategy."""
        pass
