from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseAgent(ABC):
    """Abstract base class for Reinforcement Learning agents."""

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def act(self, state: np.ndarray | torch.Tensor, deterministic: bool = False) -> Any:
        """Select an action based on the current state.

        Args:
            state: The current state observation.
            deterministic: Whether to select the action deterministically (e.g., for evaluation).

        Returns:
            The selected action.
        """
        pass

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """Update the agent's parameters based on a batch of data.

        Args:
            batch: A batch of experience data.

        Returns:
            A dictionary of metrics (e.g., loss).
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        """Save the agent's state."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """Load the agent's state."""
        pass


class BaseBuffer(ABC):
    """Abstract base class for Replay Buffers."""

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Add a transition to the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Any:
        """Sample a batch of transitions."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        pass

    @abstractmethod
    def update_priorities(self, indices: Any, priorities: Any) -> None:
        """Update priorities for prioritized replay buffer (optional)."""
        pass


class BaseExplorationStrategy(ABC):
    """Abstract base class for exploration strategies."""

    @abstractmethod
    def select_action(
        self,
        state: Any,
        action_selector: Any,
        env_action_space: Any,
    ) -> Any:
        """Select an action based on the exploration strategy."""
        pass

    @abstractmethod
    def update(self) -> None:
        """Update exploration parameters (e.g., decay epsilon)."""
        pass

    def get_epsilon(self) -> float:
        """Get current epsilon value (optional)."""
        return 0.0
