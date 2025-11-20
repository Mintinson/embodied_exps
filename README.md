# Embodied RL Experiments

A modular and extensible reinforcement learning framework for DQN-based algorithms.

## Features

- 🎯 **Modular Architecture**: Easily swap algorithms, buffers, and exploration strategies
- 🔌 **Pluggable Components**: Mix and match trainers, replay buffers, and exploration methods
- 🚀 **Multiple Algorithms**: DQN, Double DQN, easy to extend to Dueling DQN, Rainbow, etc.
- 📊 **Built-in Visualization**: Training progress plotting
- ⚙️ **Configuration Management**: YAML configs with draccus
- ✅ **Type Safe**: Comprehensive type hints throughout
- 🧪 **Testable**: Dependency injection enables easy unit testing

## Installation

This project uses `uv` for dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install development dependencies (includes ruff)
uv sync --dev
```

## Quick Start

### Train DQN on CartPole

```bash
# With default parameters
uv run python scripts/train_dqn_cartpole.py

# With custom parameters
uv run python scripts/train_dqn_cartpole.py --n_episodes 2000 --gamma 0.99

# With config file
uv run python scripts/train_dqn_cartpole.py --config configs/dqn_cartpole.yaml
```

### Train Double DQN

```bash
uv run python scripts/train_ddqn_cartpole.py
```

## Architecture

The framework is built on three core abstractions:

1. **Trainers** (`BaseTrainer`): Implement learning algorithms (DQN, DDQN, etc.)
2. **Replay Buffers** (`BaseReplayBuffer`): Manage experience storage and sampling
3. **Exploration Strategies** (`BaseExplorationStrategy`): Control action selection

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Project Structure

```
rl_models/
├── base.py                    # Abstract base classes
├── dqn_trainner.py           # DQN implementation
├── ddqn_trainer.py           # Double DQN implementation
├── exploration.py            # Exploration strategies
├── nets/
│   └── dqn_models.py        # Network architectures
└── utils/
    ├── replay_buffer.py     # Replay buffer implementations
    └── visualizer.py        # Visualization tools

scripts/
├── train_dqn_cartpole.py    # DQN training script
└── train_ddqn_cartpole.py   # Double DQN training script

configs/
├── dqn_cartpole.yaml        # DQN configuration
└── ddqn_cartpole.yaml       # DDQN configuration
```

## Usage Examples

### Basic Training Loop

```python
from rl_models import DQNTrainer, EpsilonGreedyStrategy
from rl_models.nets import DQN
from rl_models.utils import ReplayBuffer

# Setup components
model = DQN(state_size=4, action_size=2)
buffer = ReplayBuffer(max_size=10000)
exploration = EpsilonGreedyStrategy(epsilon_start=1.0, epsilon_end=0.01)

trainer = DQNTrainer(
    net=model,
    optimizer=optimizer,
    buffer=buffer,
    batch_size=64,
    gamma=0.99
)

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Select action with exploration
        action = exploration.select_action(
            state, trainer.sample_action, env.action_space
        )
        
        next_state, reward, done, _ = env.step(action)
        buffer.store(state, action, reward, next_state, done)
        
        # Learn from experience
        metrics = trainer.learn()
        state = next_state
    
    exploration.update()
```

### Using Different Components

```python
# Use prioritized replay buffer
from rl_models.utils import PrioritizedReplayBuffer
buffer = PrioritizedReplayBuffer(max_size=10000, alpha=0.6)

# Use Double DQN
from rl_models import DoubleDQNTrainer
trainer = DoubleDQNTrainer(
    net=online_net,
    target_net=target_net,
    optimizer=optimizer,
    buffer=buffer
)

# Use greedy strategy (for evaluation)
from rl_models import GreedyStrategy
exploration = GreedyStrategy()
```

## Extending the Framework

### Add a New Algorithm

```python
from rl_models.base import BaseTrainer

class MyNewAlgorithm(BaseTrainer):
    def sample_action(self, state):
        # Action selection logic
        pass
    
    def learn(self):
        # Learning algorithm
        return {"loss": loss_value}
```

### Add a New Exploration Strategy

```python
from rl_models.base import BaseExplorationStrategy

class BoltzmannExploration(BaseExplorationStrategy):
    def select_action(self, state, action_selector, env_action_space):
        # Softmax-based action selection
        pass
    
    def update(self):
        # Update temperature
        pass
```

See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for detailed refactoring rationale and migration guide.

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix
```

### Running Tests

```bash
# Coming soon: unit tests
uv run pytest
```

## Configuration

All training parameters can be configured via:

1. **Command line arguments**: `--n_episodes 2000`
2. **YAML config files**: `--config configs/my_config.yaml`
3. **Python dataclass**: Modify `TrainDQNConfig` in scripts

Example config (`configs/dqn_cartpole.yaml`):

```yaml
batch_size: 64
n_episodes: 1000
gamma: 0.95
learning_rate: 0.001
epsilon_start: 1.0
epsilon_decay: 0.995
epsilon_end: 0.01
```

## Checkpoints

Models are automatically saved to `checkpoints/` after training:

```bash
checkpoints/
├── dqn_cartpole.pth
└── ddqn_cartpole.pth
```

Load a saved model:

```python
from pathlib import Path
trainer.load(Path("checkpoints/dqn_cartpole.pth"))
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Gymnasium (with classic control environments)
- NumPy
- Matplotlib

See `pyproject.toml` for complete dependencies.

## License

MIT License

## Contributing

Contributions are welcome! The modular architecture makes it easy to add:
- New RL algorithms (Dueling DQN, Rainbow, etc.)
- New replay buffer variants (HER, n-step, etc.)
- New exploration strategies
- Additional environments
- Improved visualizations

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (DQN)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (DDQN)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

## Acknowledgments

Built with modern Python practices and design patterns for research and production use.
