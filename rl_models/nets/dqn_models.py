import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
