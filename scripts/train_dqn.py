import draccus

from rl_models.algorithms import DQN
from rl_models.common import set_seeds
from rl_models.common.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from rl_models.configs import DQNConfig
from rl_models.envs import make_env
from rl_models.runner.trainer import OffPolicyTrainer

if __name__ == "__main__":
    config = draccus.parse(DQNConfig)
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

    trainer = OffPolicyTrainer(
        agent=agent,
        env=env,
        buffer=buffer,
        config=config,
        custom_reward_fn=lambda r, d: -10.0 if d else r,
    )

    trainer.train()
