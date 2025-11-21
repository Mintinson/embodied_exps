from dataclasses import dataclass, field

from rl_models.configs.common_config import CommonConfig


@dataclass
class QNetConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])  # hidden layer sizes
    activations: str | list[str] = "relu"  # activation functions for each hidden layer


@dataclass
class DQNConfig(CommonConfig):
    """Configuration for DQN training."""

    exp_name: str = "DQN_CartPole"  # experiment name
    env_name: str = "CartPole-v1"  # environment name
    stored_dir: str = "checkpoints/dqn"  # directory to store checkpoints, logs, etc.
    qnet_config: QNetConfig = field(default_factory=QNetConfig)  # Q-network configuration
    use_prioritized_replay: bool = False  # whether to use prioritized experience replay