import draccus

from rl_models.algorithms import DDPG
from rl_models.configs import DDPGConfig
from rl_models.envs import make_env
from rl_models.runner import OffPolicyEvaluator

if __name__ == "__main__":
    config = draccus.parse(DDPGConfig)
    # set_seeds(config.seed)

    env = make_env(
        config.env_name,
        render_mode="rgb_array",
    )

    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.shape[0]  # type: ignore
    max_action = env.action_space.high[0]  # type: ignore
    agent = DDPG(state_dim, action_dim, max_action, config)

    evaluator = OffPolicyEvaluator(
        agent=agent,
        env=env,
        config=config,
    )

    evaluator.evaluate()
