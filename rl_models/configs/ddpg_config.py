from dataclasses import dataclass, field

from rl_models.configs.common_config import CommonConfig
from rl_models.core.explorations.exploration_cfgs import (
    ExplorationConfig,
    OrnsteinUhlenbeckNoiseConfig,
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
class DDPGConfig(CommonConfig):
    """Configuration for DDPG training."""

    exp_name: str = "DDPG_AntBulletEnv"  # experiment name
    env_name: str = "AntBulletEnv-v0"  # environment name
    stored_dir: str = "checkpoints/ddpg"  # directory to store checkpoints, logs, etc.
    update_target_freq: int = 100  # target network update frequency
    buffer_size: int = 1_000_000  # replay buffer size
    n_episodes: int = 1000  # number of training episodes
    use_prioritized_replay: bool = False  # whether to use prioritized experience replay
    tau: float = 0.005  # target network update rate
    actor_learning_rate: float = 1e-4  # learning rate for the actor network
    ckpt_interval: int = 100  # stored checkpoint interval
    critic_learning_rate: float = 1e-3  # learning rate for the critic network
    critic_config: CriticConfig = field(
        default_factory=CriticConfig
    )  # critic network configuration
    exploration_config: ExplorationConfig = field(
        default_factory=OrnsteinUhlenbeckNoiseConfig
    )  # exploration strategy configuration
    actor_config: ActorConfig = field(default_factory=ActorConfig)  # actor network configuration
    num_tests: int = 5  # number of test episodes
