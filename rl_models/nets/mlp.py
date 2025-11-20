from collections.abc import Sequence

import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    activation=nn.ReLU,
):
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(activation())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
