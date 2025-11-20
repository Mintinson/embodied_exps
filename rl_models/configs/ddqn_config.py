from dataclasses import dataclass

import torch


@dataclass
class QNetConfig:
    hidden_sizes: tuple = (128, 128)
    device: str = "cpu"  # or "cuda"


@dataclass
class TrainDDQNConfig:
    """Configuration for DDQN training."""

    seed: int = 42
    exp_name: str = "DDQN_CartPole"
    qnet_config: QNetConfig = QNetConfig()
    batch_size: int = 64
    n_episodes: int = 1000
    gamma: float = 0.95
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_end: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    env_name: str = "CartPole-v1"
    buffer_size: int = 2000
    stored_dir: str = "checkpoints/ddqn"
    update_target_freq: int = 10
    ckpt_interval: int = 50
    use_prioritized_replay: bool = True
