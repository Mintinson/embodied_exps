from dataclasses import dataclass

from rl_models.configs.dqn_config import DQNConfig


@dataclass
class DdoubleDQNConfig(DQNConfig):
    """Configuration for DDQN training."""

    exp_name: str = "DDQN_CartPole"  # experiment name
    stored_dir: str = "checkpoints/ddqn"  # directory to store checkpoints, logs, etc.
    update_target_freq: int = 100  # target network update frequency
    use_prioritized_replay: bool = False  # whether to use prioritized experience replay
