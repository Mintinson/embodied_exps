from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from rl_models.common import convert_to_tensor
from rl_models.configs.ddqn_config import TrainDDQNConfig
from rl_models.core.base import BaseAgent
from rl_models.nets.mlp import build_mlp


class DDQN(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TrainDDQNConfig,
        criterion: Callable | None = None,
    ):
        super().__init__(config)
        self.config: TrainDDQNConfig = config
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.target_update_freq = config.update_target_freq
        self.update_counter = 0

        self.eval_qnet = build_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=config.qnet_config.hidden_sizes,
            activation=nn.ReLU,
        ).to(self.device)

        self.target_qnet = build_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=config.qnet_config.hidden_sizes,
            activation=nn.ReLU,
        ).to(self.device)
        self.target_qnet.load_state_dict(self.eval_qnet.state_dict())

        self.optimizer = optim.Adam(self.eval_qnet.parameters(), lr=config.learning_rate)
        self.criterion = criterion or F.mse_loss

    def act(self, state: np.ndarray | torch.Tensor, deterministic: bool = False) -> int:
        if not isinstance(state, torch.Tensor):
            state = convert_to_tensor(state, device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.eval_qnet(state)

        return int(q_values.argmax().item())

    def update(self, batch: Any) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch[:5]

        states = convert_to_tensor(states, device=self.device)
        actions = convert_to_tensor(actions, torch.int64, device=self.device)
        rewards = convert_to_tensor(rewards, device=self.device)
        next_states = convert_to_tensor(next_states, device=self.device)
        dones = convert_to_tensor(dones, torch.bool, device=self.device)

        # Compute Q(s_t, a)
        q_values = self.eval_qnet(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Action selection from eval_qnet, evaluation from target_qnet
        with torch.no_grad():
            next_actions = self.eval_qnet(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_qnet(next_states)
            next_q_value = next_q_values.gather(1, next_actions).squeeze(1)
            expected_q_value = rewards + self.gamma * next_q_value * (~dones)

        loss = self.criterion(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_qnet.load_state_dict(self.eval_qnet.state_dict())

        return {"loss": loss.item(), "q_value_mean": q_value.mean().item()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.eval_qnet.load_state_dict(state_dict["eval_qnet"])
        self.target_qnet.load_state_dict(state_dict["target_qnet"])

    def state_dict(self) -> dict:
        return {
            "eval_qnet": self.eval_qnet.state_dict(),
            "target_qnet": self.target_qnet.state_dict(),
        }
