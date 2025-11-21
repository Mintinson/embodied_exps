from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from rl_models.common import convert_to_tensor
from rl_models.common.utils import LossFunction
from rl_models.configs.td3_config import TD3Config
from rl_models.core.base import BaseAgent
from rl_models.nets.mlp import build_mlp


class TD3(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        config: TD3Config,
        criterion: LossFunction | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.device = torch.device(config.device)
        self.max_action = max_action

        # Actor
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

        self.actor = lambda state: max_action * torch.tanh(self.actor_mlp(state))
        self.target_actor = lambda state: max_action * torch.tanh(self.target_actor_mlp(state))

        self.target_actor_mlp.load_state_dict(self.actor_mlp.state_dict())

        # Critic 1
        self.critic1 = build_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=config.critic_config.hidden_sizes,
            activation=config.critic_config.activations,
        ).to(self.device)
        self.target_critic1 = build_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=config.critic_config.hidden_sizes,
            activation=config.critic_config.activations,
        ).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())

        # Critic 2
        self.critic2 = build_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=config.critic_config.hidden_sizes,
            activation=config.critic_config.activations,
        ).to(self.device)
        self.target_critic2 = build_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=config.critic_config.hidden_sizes,
            activation=config.critic_config.activations,
        ).to(self.device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.tau = config.tau
        self.gamma = config.gamma
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_delay = config.policy_delay
        self.update_counter = 0

        self.actor_optimizer = optim.Adam(
            self.actor_mlp.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.critic_learning_rate,
        )

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
        self.update_counter += 1

        if len(batch) == 7:  # PrioritizedReplayBuffer
            states, actions, rewards, next_states, dones, indices, is_weight = batch
            is_weight = convert_to_tensor(is_weight, device=self.device)
        else:  # ReplayBuffer
            states, actions, rewards, next_states, dones = batch
            is_weight = 1.0

        states = convert_to_tensor(states, device=self.device)
        actions = convert_to_tensor(actions, device=self.device)
        rewards = convert_to_tensor(rewards, device=self.device)
        next_states = convert_to_tensor(next_states, device=self.device)
        dones = convert_to_tensor(dones, torch.bool, device=self.device)

        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )

        next_actions = (self.target_actor(next_states) + noise).clamp(
            -self.max_action, self.max_action
        )

        # Compute the target Q value
        target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1)).squeeze(1)
        target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1)).squeeze(1)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + (self.gamma * target_q * (~dones)).detach()

        # Get current Q estimates
        current_q1 = self.critic1(torch.cat([states, actions], dim=1)).squeeze(1)
        current_q2 = self.critic2(torch.cat([states, actions], dim=1)).squeeze(1)

        # Compute critic loss
        critic_loss = self.criterion(current_q1, target_q) + self.criterion(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {
            "critic_loss": critic_loss.item(),
            "q1": current_q1.mean().item(),
            "q2": current_q2.mean().item(),
        }

        # Delayed policy updates
        if self.update_counter % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic1(torch.cat([states, self.actor(states)], dim=1)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic1.parameters(), self.target_critic1.parameters(), strict=True
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(
                self.critic2.parameters(), self.target_critic2.parameters(), strict=True
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(
                self.actor_mlp.parameters(), self.target_actor_mlp.parameters(), strict=True
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor_mlp.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor_mlp.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.actor_mlp.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self.target_actor_mlp.load_state_dict(state_dict["target_actor"])
        self.target_critic1.load_state_dict(state_dict["target_critic1"])
        self.target_critic2.load_state_dict(state_dict["target_critic2"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
