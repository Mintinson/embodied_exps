from collections.abc import Sequence

import torch.nn as nn

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    activation: str | Sequence["str"] = "relu",
):
    layers = []
    prev_dim = input_dim
    if isinstance(activation, str):
        activation = [activation] * len(hidden_dims)
    for dim, act in zip(hidden_dims, activation, strict=False):
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(ACTIVATION_MAP[act]())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
