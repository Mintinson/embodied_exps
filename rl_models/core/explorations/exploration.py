"""Exploration strategies for RL agents.

TODO: Move this file to a more appropriate location.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from rl_models.core.base import BaseExplorationStrategy


class DummyStrategy(BaseExplorationStrategy):
    """Dummy exploration strategy that does nothing."""

    def select_action(
        self,
        state: torch.Tensor,
        action_selector: Callable[[torch.Tensor], int],
        env_action_space: Any,
    ) -> int:
        """Select action using the provided action selector."""
        return action_selector(state)

    def update(self) -> None:
        """No-op for dummy strategy."""
        pass


class ExponentGreedyStrategy(BaseExplorationStrategy):
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
        action_selector: Callable[[torch.Tensor], Any],
        env_action_space: Any,
    ) -> Any:
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


class LinearDecayEpsilonGreedyStrategy(BaseExplorationStrategy):
    """Epsilon-greedy exploration strategy with linear decay."""

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay_steps: int = 10000,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.epsilon = epsilon_start
        self.step_count = 0

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
        """Linearly decay epsilon."""
        self.step_count += 1
        decay_amount = (self.epsilon_start - self.epsilon_end) / self.decay_steps
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - decay_amount * self.step_count,
        )

    def get_epsilon(self) -> float:
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


class GaussianNoiseStrategy(BaseExplorationStrategy):
    """Gaussian noise exploration strategy for continuous action spaces."""

    def __init__(
        self,
        sigma: float = 0.1,
    ):
        self.sigma = sigma

    def select_action(
        self,
        state: torch.Tensor,
        action_selector: Callable[[torch.Tensor], np.ndarray],
        env_action_space: Any,
    ) -> np.ndarray:
        """Select action and add Gaussian noise."""
        action_dim = env_action_space.shape[0]
        max_action = np.max(np.abs(env_action_space.high))
        min_action = np.min(env_action_space.low)
        # min_ac
        action = action_selector(state)
        noise = np.random.normal(0, self.sigma, size=action_dim)
        return np.clip(action + noise, min_action, max_action)

    def update(self) -> None:
        """No-op for Gaussian noise strategy."""
        pass


class InverseTimeDecayStrategy(BaseExplorationStrategy):
    """
    Epsilon-greedy with inverse time decay.
    Formula: epsilon = epsilon_start / (1 + decay_rate * step)
    This often provides a better balance for convergence in theoretical settings.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay_rate: float = 0.001,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.step_count = 0
        self.epsilon = epsilon_start

    def select_action(
        self,
        state: torch.Tensor,
        action_selector: Callable[[torch.Tensor], int],
        env_action_space: Any,
    ) -> int:
        if np.random.rand() < self.epsilon:
            return env_action_space.sample()
        return action_selector(state)

    def update(self) -> None:
        self.step_count += 1
        decayed_epsilon = self.epsilon_start / (1 + self.decay_rate * self.step_count)
        self.epsilon = max(self.epsilon_end, decayed_epsilon)

    def get_epsilon(self) -> float:
        return self.epsilon


class CyclicalEpsilonGreedyStrategy(BaseExplorationStrategy):
    """
    Epsilon-greedy with cyclical decay (Triangular schedule).
    Useful for escaping local optima by periodically increasing exploration.
    """

    def __init__(
        self,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.01,
        cycle_steps: int = 10000,
    ):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.cycle_steps = cycle_steps
        self.step_count = 0
        self.epsilon = epsilon_max

    def select_action(
        self,
        state: torch.Tensor,
        action_selector: Callable[[torch.Tensor], int],
        env_action_space: Any,
    ) -> int:
        if np.random.rand() < self.epsilon:
            return env_action_space.sample()
        return action_selector(state)

    def update(self) -> None:
        self.step_count += 1
        # Calculate position in the cycle [0, 1]
        cycle_progress = (self.step_count % self.cycle_steps) / self.cycle_steps

        if cycle_progress < 0.5:
            # Decay phase (1.0 -> 0.01)
            factor = 1.0 - (cycle_progress * 2)
        else:
            # Rise phase (0.01 -> 1.0)
            factor = (cycle_progress - 0.5) * 2

        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * factor

    def get_epsilon(self) -> float:
        return self.epsilon


class OrnsteinUhlenbeckNoiseStrategy(BaseExplorationStrategy):
    """
    Ornstein-Uhlenbeck process for temporally correlated noise.
    Essential for continuous control tasks (e.g., DDPG, TD3) where
    momentum in exploration is desired.
    """

    def __init__(
        self,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        initial_noise: Sequence[float] | None = None,
    ):
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.initial_noise = initial_noise

        self.noise = np.zeros(1)

    def select_action(
        self,
        state: torch.Tensor,
        action_selector: Callable[[torch.Tensor], np.ndarray],
        env_action_space: Any,
    ) -> np.ndarray:
        """Select action and add OU noise."""
        action = action_selector(state)

        action_dim = env_action_space.shape[0]
        max_action = env_action_space.high
        min_action = env_action_space.low
        noise = (
            self.noise
            + self.theta * (0 - self.noise) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=action_dim)
        )
        self.noise = noise

        return np.clip(action + noise, min_action, max_action)

    def update(self) -> None:
        """No-op for OU noise, but could implement sigma decay here if needed."""
        pass
