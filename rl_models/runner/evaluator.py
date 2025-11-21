from typing import Any

import numpy as np
from gymnasium import Env

from rl_models.core.base import BaseAgent
from rl_models.runner.recorder import Recorder


class OffPolicyEvaluator:
    """Trainer for off-policy algorithms (DQN, DDPG, SAC, etc.)."""

    def __init__(
        self,
        agent: BaseAgent,
        env: Env,
        config: Any,
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.recorder = Recorder(config, is_training=False)

    def evaluate(self) -> None:
        """Execute the evaluation loop."""
        self.recorder.logger.info(f"Starting evaluation {self.config.env_name}")
        self.agent.eval()
        self.agent.load_state_dict(self.recorder.load_model(self.config.ckpt_path))

        # rewards = []
        best_rewards = float("-inf")
        best_frames = []

        for e in range(self.config.num_tests):
            frames = []
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False

            while not done:
                # Action selection
                action = self.agent.act(state=state, deterministic=True)

                # Environment step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                frame: np.ndarray = self.env.render()  # pyright: ignore[reportAssignmentType]
                if frame.dtype != np.uint8:
                    frame = 255 * (1.0 * frame - frame.min()) / (frame.max() - frame.min())
                    frame = frame.astype(np.uint8)
                frames.append(frame)

                state = next_state
                total_reward += float(reward)

            self.recorder.logger.info(f"Episode: {e + 1}, Score: {total_reward:.2f}.")
            if total_reward > best_rewards:
                best_rewards = total_reward
                best_frames = frames
        self.recorder.logger.info(f"Evaluation finished. Best Score: {best_rewards:.2f}.")
        self.recorder.save_gif(best_frames)
