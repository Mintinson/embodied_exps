from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from rl_models.common import convert_to_tensor
from rl_models.common.utils import LossFunction
from rl_models.configs.ddpg_config import DDPGConfig
from rl_models.core.base import BaseAgent
from rl_models.nets.mlp import build_mlp


class DDPG(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        config: DDPGConfig,
        criterion: LossFunction | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.device = torch.device(config.device)
        self.actor_mlp = build_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=config.actor_config.hidden_sizes,
            activation=config.actor_config.activations,
        ).to(self.device)
        self.target_actor_mlp = build_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=config.actor_config.hidden_sizes,
            activation=config.actor_config.activations,
        ).to(self.device)

        self.target_actor = lambda state: max_action * torch.tanh(self.target_actor_mlp(state))

        self.actor = lambda state: max_action * torch.tanh(self.actor_mlp(state))
        self.critic = build_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=config.critic_config.hidden_sizes,
            activation=config.critic_config.activations,
        ).to(self.device)
        self.target_critic = build_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=config.critic_config.hidden_sizes,
            activation=config.critic_config.activations,
        ).to(self.device)
        self.target_actor_mlp.load_state_dict(self.actor_mlp.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.tau = config.tau
        self.gamma = config.gamma
        self.target_update_freq = config.update_target_freq
        self.update_counter = 0

        self.actor_optimizer = optim.Adam(
            self.actor_mlp.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate)

        self.criterion = criterion or F.mse_loss

    def act(self, state: np.ndarray | torch.Tensor, deterministic: bool = False) -> np.ndarray:
        if not isinstance(state, torch.Tensor):
            state = convert_to_tensor(state, device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def update(self, batch: Any) -> dict[str, Any]:
        if len(batch) == 7:  #  PrioritizedReplayBuffer
            states, actions, rewards, next_states, dones, indices, is_weight = batch
            is_weight = convert_to_tensor(is_weight, device=self.device)
        else:  #  ReplayBuffer
            states, actions, rewards, next_states, dones = batch
            is_weight = 1.0

        states = convert_to_tensor(states, device=self.device)
        actions = convert_to_tensor(actions, torch.int64, device=self.device)
        rewards = convert_to_tensor(rewards, device=self.device)
        next_states = convert_to_tensor(next_states, device=self.device)
        dones = convert_to_tensor(dones, torch.bool, device=self.device)

        # critic loss
        target_actions = self.target_actor(next_states)
        target_q = self.target_critic(torch.cat([next_states, target_actions], dim=1)).squeeze(1)
        target_q = rewards + (self.gamma * target_q * (~dones)).detach()

        current_q = self.critic(torch.cat([states, actions], dim=1)).squeeze(1)
        critic_loss_elementwise = self.criterion(current_q, target_q, reduction="none")
        critic_loss = (critic_loss_elementwise * is_weight).mean()

        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()

        # actor loss
        actor_loss = -self.critic(torch.cat([states, self.actor(states)], dim=1)).mean()

        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # soft update
        self.soft_update(self.actor_mlp, self.target_actor_mlp)
        self.soft_update(self.critic, self.target_critic)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "td_errors": td_errors,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.actor_mlp.load_state_dict(state_dict["actor"])
        self.target_actor_mlp.load_state_dict(state_dict["target_actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])

    def state_dict(self) -> dict:
        return {
            "actor": self.actor_mlp.state_dict(),
            "target_actor": self.target_actor_mlp.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }

    def soft_update(self, net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        for param, target_param in zip(net.parameters(), target_net.parameters(), strict=False):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
