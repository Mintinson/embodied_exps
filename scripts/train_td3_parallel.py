import draccus

from rl_models.algorithms.td3 import TD3
from rl_models.common.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from rl_models.common.utils import set_seeds
from rl_models.configs.td3_config import TD3Config
from rl_models.envs import SubprocVecEnv, make_env
from rl_models.runner.trainer import OffPolicyTrainer

if __name__ == "__main__":
    config = draccus.parse(TD3Config)
    set_seeds(config.seed)

    env = make_env(
        config.env_name,
        render_mode=getattr(config, "render_mode", None),
    )
    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.shape[0]  # type: ignore
    max_action = env.action_space.high[0]  # type: ignore

    env = SubprocVecEnv(config.env_name, render_mode=getattr(config, "render_mode", None))

    agent = TD3(state_dim, action_dim, max_action, config)

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
