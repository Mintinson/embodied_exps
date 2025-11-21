import random
from typing import Protocol

import numpy as np
import torch


class LossFunction(Protocol):
    def __call__(
        self, input: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean"
    ) -> torch.Tensor: ...


def convert_to_tensor(
    data: np.ndarray | list | torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert numpy array or list to PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype=dtype)
    else:
        tensor = torch.tensor(data, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
