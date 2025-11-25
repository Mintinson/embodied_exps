from dataclasses import dataclass, field

import draccus

from rl_models.core.base import BaseExplorationStrategy
from rl_models.core.explorations.exploration import (
    CyclicalEpsilonGreedyStrategy,
    DummyStrategy,
    ExponentGreedyStrategy,
    GaussianNoiseStrategy,
    InverseTimeDecayStrategy,
    LinearDecayEpsilonGreedyStrategy,
    OrnsteinUhlenbeckNoiseStrategy,
)


@dataclass
class ExplorationConfig(draccus.ChoiceRegistry):
    pass


@dataclass
class DummyConfig(ExplorationConfig):
    pass


@dataclass
class ExponentGreedyConfig(ExplorationConfig):
    """Epsilon-greedy exploration strategy configuration."""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995


@dataclass
class LinearDecayEpsilonGreedyConfig(ExplorationConfig):
    """Linear decay epsilon-greedy exploration strategy configuration."""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    decay_steps: int = 10000


@dataclass
class GaussianNoiseConfig(ExplorationConfig):
    """Gaussian noise exploration strategy configuration."""

    sigma: float = 0.1


@dataclass
class InverseTimeDecayConfig(ExplorationConfig):
    """Inverse time decay exploration strategy configuration."""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    decay_rate: float = 0.001


@dataclass
class CyclicalEpsilonConfig(ExplorationConfig):
    """Cyclical epsilon exploration strategy configuration."""

    epsilon_max: float = 0.01
    epsilon_min: float = 1.0
    cycle_steps: int = 1000


@dataclass
class OrnsteinUhlenbeckNoiseConfig(ExplorationConfig):
    """Ornstein-Uhlenbeck noise exploration strategy configuration."""

    sigma: float = 0.2
    theta: float = 0.15
    dt: float = 1e-2
    initial_noise: list[float] = field(default_factory=lambda: [])


@dataclass
class ExplorationTuple:
    config: type[ExplorationConfig]
    strategy_class: type[BaseExplorationStrategy]


EXPLORATION_MAP = {
    # "dummy": (DummyConfig, DummyStrategy),
    "dummy": ExplorationTuple(DummyConfig, DummyStrategy),
    "exponent_greedy": ExplorationTuple(ExponentGreedyConfig, ExponentGreedyStrategy),
    "linear_decay_epsilon": ExplorationTuple(
        LinearDecayEpsilonGreedyConfig, LinearDecayEpsilonGreedyStrategy
    ),
    "gaussian_noise": ExplorationTuple(GaussianNoiseConfig, GaussianNoiseStrategy),
    "inverse_time_decay": ExplorationTuple(InverseTimeDecayConfig, InverseTimeDecayStrategy),
    "cyclical": ExplorationTuple(CyclicalEpsilonConfig, CyclicalEpsilonGreedyStrategy),
    "ornstein": ExplorationTuple(OrnsteinUhlenbeckNoiseConfig, OrnsteinUhlenbeckNoiseStrategy),
    # add more strategies here as needed
}


for key, exploration_tuple in EXPLORATION_MAP.items():
    ExplorationConfig.register_subclass(key, exploration_tuple.config)
