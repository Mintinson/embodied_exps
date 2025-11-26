from dataclasses import dataclass, field

from rl_models.configs.common_config import CommonConfig
from rl_models.core.explorations.exploration_cfgs import (
    ExplorationConfig,
    GaussianNoiseConfig,
)


@dataclass
class CriticConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [400, 300])  # hidden layer sizes
    activations: str | list[str] = "relu"  # activation functions for each hidden layer


@dataclass
class ActorConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [400, 300])  # hidden layer sizes
    activations: str | list[str] = "relu"  # activation functions for each hidden layer


@dataclass
class TD3Config(CommonConfig):
    """Configuration for TD3 training."""

    exp_name: str = "TD3_AntBulletEnv"  # experiment name
    env_name: str = "AntBulletEnv-v0"  # environment name
    stored_dir: str = "checkpoints/td3"  # directory to store checkpoints, logs, etc.
    buffer_size: int = 1_000_000  # replay buffer size
    n_episodes: int = 1000  # number of training episodes
    use_prioritized_replay: bool = False  # whether to use prioritized experience replay
    tau: float = 0.005  # target network update rate
    actor_learning_rate: float = 1e-3  # learning rate for the actor network
    critic_learning_rate: float = 1e-3  # learning rate for the critic network
    ckpt_interval: int = 10  # stored checkpoint interval

    # TD3 specific parameters
    policy_noise: float = 0.2  # Noise added to target policy during critic update
    noise_clip: float = 0.5  # Range to clip target policy noise
    policy_delay: int = 2  # Frequency of delayed policy updates
    exploration_noise: float = 0.1  # Standard deviation for exploration noise

    critic_config: CriticConfig = field(
        default_factory=CriticConfig
    )  # critic network configuration
    actor_config: ActorConfig = field(default_factory=ActorConfig)  # actor network configuration
    num_tests: int = 5  # number of test episodes
    exploration_config: ExplorationConfig = field(
        default_factory=GaussianNoiseConfig
    )  # exploration strategy configuration
