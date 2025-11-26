import dataclasses
import datetime
import json
from collections.abc import Sequence
from pathlib import Path

import draccus
import torch

from rl_models.common.logger import get_logger


class Recorder:
    def __init__(self, config, is_training: bool) -> None:
        self.config = config
        if is_training:
            self.target_dir = Path(config.stored_dir) / datetime.datetime.now().strftime(
                "%Y%m%d-%H%M"
            )
            self.config.stored_dir = str(self.target_dir)
            self.target_dir.mkdir(parents=True, exist_ok=True)
            cfg_dict = draccus.encode(config)
            with open(self.target_dir / "config.json", "w") as f:
                json.dump(cfg_dict, f, indent=4)
        else:
            self.target_dir = Path(config.stored_dir)
            # self.target_dir = Path(config.stored_dir) / "evaluation"
        log_name = "training" if is_training else "evaluation"
        self.logger = get_logger(config.exp_name, self.target_dir / f"{log_name}.log")
        self.logger.info(
            f"Experiment configuration: {json.dumps(dataclasses.asdict(config), indent=4)}"
        )

    # def start_train(self) -> None:

    def load_model(self, path: Path, is_full=False) -> dict:
        if is_full:
            target_path = path
        else:
            target_path = self.target_dir / path
        self.logger.info(f"Loaded model from {target_path}")
        return torch.load(target_path)

    def save_model(self, state_dict: dict, ckpt_path: str) -> None:
        full_path = self.target_dir / ckpt_path
        # full_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, full_path)
        self.logger.info(f"Saved model to {full_path}")

    def _smooth(self, data, weight=0.9):
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def plot_rewards(self, rewards: Sequence[float], is_training: bool = True) -> None:
        import matplotlib.pyplot as plt

        plt.plot(rewards, label="rewards")
        plt.plot(self._smooth(rewards), label="smoothed")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(
            ("Training Rewards" if is_training else "Evaluation Rewards")
            + f" - {self.config.exp_name}"
        )
        plt.savefig(self.target_dir / "rewards.png")
        plt.legend()
        plt.show()

    def save_gif(self, frames: Sequence) -> None:
        import imageio

        gif_path = self.target_dir / "best_episode.gif"
        imageio.mimsave(gif_path, frames, fps=30)  # pyright: ignore[reportArgumentType, reportCallIssue]
        self.logger.info(f"Saved GIF to {gif_path}")