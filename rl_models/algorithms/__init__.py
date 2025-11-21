# from dq
from .ddpg import DDPG
from .ddqn import DDQN
from .dqn import DQN
from .td3 import TD3

__all__ = ["DQN", "DDQN", "DDPG", "TD3"]
