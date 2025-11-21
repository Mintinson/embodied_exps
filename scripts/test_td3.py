import draccus

from rl_models.algorithms.td3 import TD3
from rl_models.configs.td3_config import TD3Config
from rl_models.envs import make_env
from rl_models.runner.evaluator import OffPolicyEvaluator

if __name__ == "__main__":
    config = draccus.parse(TD3Config)

    env = make_env(
        config.env_name,
        render_mode="rgb_array",
    )

    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.shape[0]  # type: ignore
    max_action = env.action_space.high[0]  # type: ignore
    agent = TD3(state_dim, action_dim, max_action, config)

    evaluator = OffPolicyEvaluator(
        agent=agent,
        env=env,
        config=config,
    )

    evaluator.evaluate()
