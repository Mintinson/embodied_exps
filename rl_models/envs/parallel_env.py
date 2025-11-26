import multiprocessing as mp

import numpy as np

from rl_models.envs.env_plant import make_env


def worker(remote, parent_remote, env_name: str, render_mode: str | None = None):
    parent_remote.close()
    env = make_env(env_name, render_mode=render_mode)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                next_state, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    next_state, info = env.reset()
                remote.send((next_state, reward, terminated, truncated, info))
            elif cmd == "reset":
                state, info = env.reset(seed=data)
                remote.send((state, info))
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        env.close()


class SubprocVecEnv:
    """Creates a vectorized environment that runs multiple environments in parallel subprocesses."""

    def __init__(self, env_name, render_mode: str | None = None, num_envs: int = 4):
        self.closed = False
        self.num_envs = num_envs

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)], strict=False)
        self.ps = [
            mp.Process(target=worker, args=(work_remote, remote, env_name, render_mode))
            for (work_remote, remote) in zip(self.work_remotes, self.remotes, strict=False)
        ]

        for p in self.ps:
            p.daemon = True  # 主进程结束时子进程自动结束
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # get the information of observation_space and action_space from one of the environments
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions, strict=False):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, terminated, truncated, infos = zip(*results, strict=False)
        return np.stack(obs), np.stack(rews), np.stack(terminated), np.stack(truncated), infos

    def reset(self, seeds=None):
        if seeds is None:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds, strict=False):
            remote.send(("reset", seed))

        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results, strict=False)

        return np.stack(obs), infos

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True
