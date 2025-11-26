import gymnasium as gym
import pybullet_envs_gymnasium  # noqa: F401


def make_env(env_name: str, render_mode: str | None = None):
    """Create and return a Gymnasium environment."""
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)
    return env
