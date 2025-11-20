"""Replay buffer implementations."""

import random
from collections import deque
from typing import Any

import numpy as np

from rl_models.common.sum_tree import SumTree
from rl_models.core.base import BaseBuffer


class ReplayBuffer(BaseBuffer):
    """Standard uniform replay buffer."""

    def __init__(self, max_size: int) -> None:
        self.memory: deque = deque(maxlen=max_size)

    def add(self, *args, **kwargs) -> None:
        """Store a transition in the buffer."""
        if len(args) == 1 and isinstance(args[0], tuple):
            self.memory.append(args[0])
        else:
            self.memory.append(args)

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions uniformly."""
        batch = random.sample(self.memory, batch_size)
        return tuple(map(np.stack, zip(*batch, strict=False)))

    def update_priorities(self, indices: Any, priorities: Any) -> None:
        """No-op for uniform replay buffer."""
        pass

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.memory)


class PrioritizedReplayBuffer(BaseBuffer):
    """Prioritized experience replay buffer."""

    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ) -> None:
        self.tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01

    def add(self, *args, **kwargs) -> None:
        """Store a transition with max priority."""
        if len(args) == 1 and isinstance(args[0], tuple):
            experience = args[0]
        else:
            experience = args

        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, experience)

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch based on priorities."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch, strict=False)
        return (
            np.stack(states),
            np.stack(actions),
            np.stack(rewards),
            np.stack(next_states),
            np.stack(dones),
            indices,
            is_weight,
        )

    def update_priorities(self, indices: Any, priorities: Any) -> None:
        """Update priorities of sampled transitions."""
        errors = priorities
        for idx, error in zip(indices, errors, strict=False):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
