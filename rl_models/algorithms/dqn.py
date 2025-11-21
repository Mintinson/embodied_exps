from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from rl_models.common import convert_to_tensor
from rl_models.configs.dqn_config import DQNConfig
from rl_models.core.base import BaseAgent
from rl_models.nets.mlp import build_mlp


class DQN(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DQNConfig,
        criterion: Callable | None = None,
    ):
        super().__init__(config)
        self.config: DQNConfig = config
        self.device = torch.device(config.device)
        self.gamma = config.gamma

        self.qnet = build_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=config.qnet_config.hidden_sizes,
            activation=config.qnet_config.activations,
        ).to(self.device)

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=config.learning_rate)
        self.criterion = criterion or F.mse_loss

    def act(self, state: np.ndarray | torch.Tensor, deterministic: bool = False) -> int:
        if not isinstance(state, torch.Tensor):
            state = convert_to_tensor(state, device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.qnet(state)

        return int(q_values.argmax().item())

    def update(self, batch: Any) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch[:5]

        states = convert_to_tensor(states, device=self.device)
        actions = convert_to_tensor(actions, torch.int64, device=self.device)
        rewards = convert_to_tensor(rewards, device=self.device)
        next_states = convert_to_tensor(next_states, device=self.device)
        dones = convert_to_tensor(dones, torch.bool, device=self.device)

        # Compute Q(s_t, a)
        q_values = self.qnet(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_q_values = self.qnet(next_states)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + self.gamma * next_q_value * (~dones)

        loss = self.criterion(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "q_value_mean": q_value.mean().item()}

    def state_dict(self) -> dict:
        return self.qnet.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.qnet.load_state_dict(state_dict)
