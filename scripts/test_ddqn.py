import draccus

from rl_models.algorithms import DDQN
from rl_models.configs import DdoubleDQNConfig
from rl_models.envs import make_env
from rl_models.runner import OffPolicyEvaluator

if __name__ == "__main__":
    config = draccus.parse(DdoubleDQNConfig)
    print(config)
    # set_seeds(config.seed)

    env = make_env(
        config.env_name,
        render_mode="rgb_array",
    )

    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.n  # type: ignore
    agent = DDQN(state_dim, action_dim, config)

    evaluator = OffPolicyEvaluator(
        agent=agent,
        env=env,
        config=config,
    )

    evaluator.evaluate()
