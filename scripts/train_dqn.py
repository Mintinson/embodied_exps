import random

import draccus
import numpy as np
import torch

from rl_models.algorithms import DQN
from rl_models.common.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from rl_models.configs import TrainDQNConfig
from rl_models.envs import make_env
from rl_models.exploration import EpsilonGreedyStrategy
from rl_models.runner.trainer import OffPolicyTrainer


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    config = draccus.parse(TrainDQNConfig)
    set_seeds(config.seed)

    env = make_env(
        config.env_name,
        render_mode=getattr(config, "render_mode", None),
    )

    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.n  # type: ignore
    agent = DQN(state_dim, action_dim, config)

    if config.use_prioritized_replay:
        buffer = PrioritizedReplayBuffer(max_size=config.buffer_size)
    else:
        buffer = ReplayBuffer(max_size=config.buffer_size)

    exploration_strategy = EpsilonGreedyStrategy(
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
    )

    trainer = OffPolicyTrainer(
        agent=agent,
        env=env,
        buffer=buffer,
        exploration_strategy=exploration_strategy,
        config=config,
    )

    trainer.train()
