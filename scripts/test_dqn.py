import draccus

from rl_models.algorithms import DQN
from rl_models.configs import DQNConfig
from rl_models.envs import make_env
from rl_models.runner import OffPolicyEvaluator

if __name__ == "__main__":
    config = draccus.parse(DQNConfig)
    print(config)

    env = make_env(
        config.env_name,
        render_mode="rgb_array",
    )

    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.n  # type: ignore
    agent = DQN(state_dim, action_dim, config)

    evaluator = OffPolicyEvaluator(
        agent=agent,
        env=env,
        config=config,
    )

    evaluator.evaluate()
