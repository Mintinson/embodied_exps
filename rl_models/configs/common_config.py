from dataclasses import dataclass, field

import torch

from rl_models.core.explorations.exploration_cfgs import ExplorationConfig, ExponentGreedyConfig


@dataclass
class OptimizerConfig:
    name: str = "adam"  # optimizer name
    learning_rate: float = 1e-3  # learning rate
    weight_decay: float = 0.0  # weight decay


@dataclass
class CommonConfig:
    """Common configuration for RL models."""

    exp_name: str  # experiment name
    stored_dir: str  # directory to store checkpoints, logs, etc.
    env_name: str  # environment name
    seed: int = 42  # training seed
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # device to use
    batch_size: int = 64  # mini-batch size
    n_episodes: int = 1000  # number of training episodes
    gamma: float = 0.95  # discount factor
    learning_rate: float = 1e-3  # learning rate
    epsilon_start: float = 1.0  # initial exploration rate
    epsilon_decay: float = 0.995  # exploration rate decay
    epsilon_end: float = 0.01  # minimum exploration rate
    buffer_size: int = 2000  # replay buffer size
    ckpt_interval: int = 50  # stored checkpoint interval

    ckpt_path: str = "model_last.pth"  # checkpoint path for evaluation
    num_tests: int = 50  # number of test episodes

    is_learn_per_step: bool = True  # learn the agent per step
    learn_per_unit: int = 1  # learning frequency per unit

    exploration_config: ExplorationConfig = field(default_factory=ExponentGreedyConfig)