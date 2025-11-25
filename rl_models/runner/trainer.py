from collections.abc import Callable
from typing import Any

import numpy as np
from gymnasium import Env

from rl_models.core.base import BaseAgent, BaseBuffer, BaseExplorationStrategy
from rl_models.core.explorations.exploration_cfgs import EXPLORATION_MAP, ExplorationConfig
from rl_models.runner.recorder import Recorder


class OffPolicyTrainer:
    """Trainer for off-policy algorithms (DQN, DDPG, SAC, etc.)."""

    def __init__(
        self,
        agent: BaseAgent,
        env: Env,
        buffer: BaseBuffer,
        config: Any,
        custom_reward_fn: Callable[[float, bool], float] | None = None,
    ):
        self.config = config
        self.agent = agent
        self.env = env
        self.buffer = buffer

        StrategyClass = EXPLORATION_MAP[
            ExplorationConfig.get_choice_name(type(config.exploration_config))
        ].strategy_class

        self.exploration_strategy: BaseExplorationStrategy = StrategyClass(
            **vars(config.exploration_config)
        )
        self.custom_reward_fn = custom_reward_fn or (lambda r, d: r)
        self.recorder = Recorder(config, is_training=True)

    def train(self) -> None:
        """Execute the training loop."""
        self.recorder.logger.info(f"Starting training {self.config.env_name}")

        rewards = []
        for e in range(self.config.n_episodes):
            episode_seed = self.config.seed + e
            state, _ = self.env.reset(seed=episode_seed)
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False
            step = 0

            while not done:
                # Action selection
                action = self.exploration_strategy.select_action(
                    state, self.agent.act, self.env.action_space
                )

                # Environment step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Custom reward shaping or casting
                reward = self.custom_reward_fn(float(reward), done)

                next_state = np.array(next_state, dtype=np.float32)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]

                # Store experience
                self.buffer.add(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                step += 1

                # Update agent
                if self.config.is_learn_per_step:
                    if step % self.config.learn_per_unit == 0:
                        self._replay_learn()

            if not self.config.is_learn_per_step:
                if (e + 1) % self.config.learn_per_unit == 0:
                    self._replay_learn()

            # End of episode
            self.exploration_strategy.update()
            rewards.append(total_reward)

            self.recorder.logger.info(
                f"Episode: {e + 1}/{self.config.n_episodes}, "
                f"Score: {total_reward:.2f}, "
                f"Steps: {step}, "
                f"Epsilon: {self.exploration_strategy.get_epsilon():.3f}"
            )

            # Checkpointing
            if hasattr(self.config, "ckpt_interval") and (e + 1) % self.config.ckpt_interval == 0:
                self.recorder.save_model(self.agent.state_dict(), f"model_ep{e + 1}.pth")

        # self.recorder.save_model(self.agent)

        self.recorder.logger.info("Training finished.")
        self.recorder.save_model(self.agent.state_dict(), "model_last.pth")
        self.recorder.plot_rewards(rewards)

    def _replay_learn(self):
        if len(self.buffer) >= self.config.batch_size:
            batch = self.buffer.sample(self.config.batch_size)
            metrics = self.agent.update(batch)

            # Handle prioritized replay updates if needed
            if "td_errors" in metrics and hasattr(self.buffer, "update_priorities"):
                if isinstance(batch, tuple) and len(batch) > 5:
                    indices = batch[5]
                    self.buffer.update_priorities(indices, metrics["td_errors"])
