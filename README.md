# Embodied RL Experiments

一个模块化且可扩展的强化学习框架，支持离散和连续控制的多种深度强化学习算法。

## 特性

- 🎯 **模块化架构**: 轻松切换算法、缓冲区和探索策略
- 🔌 **可插拔组件**: 混合搭配训练器、回放缓冲区和探索方法
- 🚀 **多种算法**: DQN、Double DQN、DDPG、TD3，易于扩展到 Dueling DQN、Rainbow、SAC、PPO 等
- 📊 **内置可视化**: 训练进度图表
- ⚙️ **配置管理**: 使用 draccus 进行配置管理
- ✅ **类型安全**: 全面的类型提示
- 🧪 **可测试**: 依赖注入使单元测试变得简单

## 安装

本项目使用 `uv` 进行依赖管理：

```bash
# 如果尚未安装 uv，先安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 安装开发依赖（包括 ruff）
uv sync --dev
```

也支持使用 `conda` 管理依赖：

```bash
conda env create -f environment.yml
conda activate embodied-exps
```

或者直接使用 `pip`：

```bash
pip install -r requirements.txt
```

## 快速开始

> 注意：如果不是使用 `uv` 管理项目的话，将下述命令中的 `uv run` 替换成 `python` 即可。

### 训练离散控制算法（DQN 系列）

```bash
# 训练 DQN（CartPole 环境）
uv run scripts/train_dqn.py

# 使用自定义参数
uv run scripts/train_dqn.py --n_episodes 2000 --gamma 0.99

# 训练 Double DQN
uv run scripts/train_ddqn.py

# 测试训练好的模型
uv run scripts/test_dqn.py --config_path checkpoints/dqn/<timestamp>/config.json --ckpt_path model_ep800.pth
```

### 训练连续控制算法（DDPG/TD3）

```bash
# 训练 DDPG（AntBulletEnv 环境）
uv run scripts/train_ddpg.py

# 训练 TD3（Pendulum 环境）
uv run scripts/train_td3.py --env_name Pendulum-v1 --n_episodes 100

# 测试训练好的模型
uv run scripts/test_td3.py --config_path checkpoints/td3/<timestamp>/config.json
```

## 架构

框架建立在以下核心抽象之上：

1. **智能体** (`BaseAgent`): 实现学习算法（DQN、DDQN、DDPG、TD3 等）
2. **回放缓冲区** (`BaseBuffer`): 管理经验存储和采样
3. **探索策略** (`BaseExplorationStrategy`): 控制动作选择
4. **训练器** (`OffPolicyTrainer`): 通用的离策略训练循环
5. **评估器** (`OffPolicyEvaluator`): 模型评估和可视化

## 项目结构

```
rl_models/
├── core/
│   └── base.py                   # 抽象基类（BaseAgent, BaseBuffer, BaseExplorationStrategy）
├── algorithms/
│   ├── dqn.py                    # DQN 实现
│   ├── ddqn.py                   # Double DQN 实现
│   ├── ddpg.py                   # DDPG 实现
│   └── td3.py                    # TD3 实现
├── common/
│   ├── replay_buffer.py          # 回放缓冲区实现（普通/优先级）
│   ├── sum_tree.py               # 优先级回放使用的 SumTree
│   ├── logger.py                 # 日志工具
│   └── utils.py                  # 工具函数
├── configs/
│   ├── common_config.py          # 通用配置
│   ├── dqn_config.py             # DQN 配置
│   ├── ddqn_config.py            # DDQN 配置
│   ├── ddpg_config.py            # DDPG 配置
│   └── td3_config.py             # TD3 配置
├── nets/
│   ├── dqn_models.py             # DQN 网络架构
│   ├── dqpg_models.py            # DDPG/TD3 网络架构
│   └── mlp.py                    # 通用 MLP 构建器
├── runner/
│   ├── trainer.py                # 通用离策略训练器
│   ├── evaluator.py              # 模型评估器
│   └── recorder.py               # 日志记录和模型保存
└── exploration.py                # 探索策略（Epsilon-Greedy, Gaussian Noise 等）

scripts/
├── train_dqn.py                  # DQN 训练脚本
├── train_ddqn.py                 # Double DQN 训练脚本
├── train_ddpg.py                 # DDPG 训练脚本
├── train_td3.py                  # TD3 训练脚本
├── test_dqn.py                   # DQN 测试脚本
├── test_ddqn.py                  # Double DQN 测试脚本
├── test_ddpg.py                  # DDPG 测试脚本
└── test_td3.py                   # TD3 测试脚本

checkpoints/                      # 模型检查点和日志
```

## 使用示例

### 基本训练流程

```python
import draccus
from rl_models.algorithms import DQN
from rl_models.common.replay_buffer import ReplayBuffer
from rl_models.configs import DQNConfig
from rl_models.envs import make_env
from rl_models.exploration import EpsilonGreedyStrategy
from rl_models.runner.trainer import OffPolicyTrainer

# 解析配置
config = draccus.parse(DQNConfig)

# 创建环境
env = make_env(config.env_name)

# 创建智能体
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, action_dim, config)

# 创建回放缓冲区
buffer = ReplayBuffer(max_size=config.buffer_size)

# 创建探索策略
exploration_strategy = EpsilonGreedyStrategy(
    epsilon_start=config.epsilon_start,
    epsilon_end=config.epsilon_end,
    epsilon_decay=config.epsilon_decay,
)

# 创建训练器
trainer = OffPolicyTrainer(
    agent=agent,
    env=env,
    buffer=buffer,
    exploration_strategy=exploration_strategy,
    config=config,
)

# 开始训练
trainer.train()
```

### 使用不同的组件

```python
# 使用优先级回放缓冲区
from rl_models.common.replay_buffer import PrioritizedReplayBuffer
buffer = PrioritizedReplayBuffer(max_size=config.buffer_size)

# 使用 TD3 算法（连续控制）
from rl_models.algorithms import TD3
from rl_models.exploration import GaussianNoiseStrategy

agent = TD3(state_dim, action_dim, max_action, config)
exploration_strategy = GaussianNoiseStrategy(
    action_dim=action_dim,
    max_action=max_action,
    sigma=0.1,
)

# 使用贪婪策略（用于评估）
from rl_models.exploration import GreedyStrategy
exploration = GreedyStrategy()
```

## 扩展框架

### 添加新算法

```python
from rl_models.core.base import BaseAgent

class MyNewAlgorithm(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, config):
        super().__init__(config)
        # 初始化网络、优化器等
    
    def act(self, state, deterministic=False):
        # 动作选择逻辑
        pass
    
    def update(self, batch):
        # 学习算法
        return {"loss": loss_value}
    
    def state_dict(self):
        # 返回需要保存的参数
        pass
    
    def load_state_dict(self, state_dict):
        # 加载参数
        pass
```

### 添加新探索策略

```python
from rl_models.core.base import BaseExplorationStrategy

class BoltzmannExploration(BaseExplorationStrategy):
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def select_action(self, state, action_selector, env_action_space):
        # 基于 Softmax 的动作选择
        pass
    
    def update(self):
        # 更新温度参数
        pass
```

## 开发

### 代码质量

```bash
# 格式化代码
uv run ruff format .

# 代码检查
uv run ruff check .

# 自动修复可修复的问题
uv run ruff check . --fix
```

## 配置

所有训练参数可以通过以下方式配置：

1. **命令行参数**: `uv run script.py --n_episodes 2000 --gamma 0.99`
3. **配置文件**： `uv run script.py --config_path your_cfg.json` 也可以是 `.yaml` 文件  
2. **Python dataclass**: 在 `rl_models/configs/` 中修改对应的配置类

配置示例（DQN）：

```python
@dataclass
class DQNConfig(CommonConfig):
    exp_name: str = "DQN_CartPole"
    env_name: str = "CartPole-v1"
    batch_size: int = 64
    n_episodes: int = 1000
    gamma: float = 0.95
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_end: float = 0.01
    buffer_size: int = 2000
    use_prioritized_replay: bool = True
```

## 检查点

训练后模型会自动保存到 `checkpoints/` 目录：

```bash
checkpoints/
├── dqn/
│   └── 20251121-1642/
│       ├── config.json
│       ├── model_ep800.pth
│       └── model_last.pth
├── ddqn/
├── ddpg/
└── td3/
    └── 20251121-1929/
        ├── config.json
        └── model_last.pth
```

检验模型：

```bash
# if --ckpt_path doesn't specify, it will choose model_last.pth in the config_path directory
uv run scripts/test_xxx.py --config_path your_json_yaml_path --ckpt_path your_ckpt_path
```

## 支持的算法

### 离散控制（Discrete Action Space）
- **DQN** (Deep Q-Network): 基础的深度 Q 学习算法
- **Double DQN**: 使用双网络减少 Q 值高估

### 连续控制（Continuous Action Space）
- **DDPG** (Deep Deterministic Policy Gradient): 确定性策略梯度算法
- **TD3** (Twin Delayed DDPG): 改进的 DDPG，使用双 Critic 和延迟策略更新

### 探索策略
- **Epsilon-Greedy**: 用于离散动作空间（DQN 系列）
- **Gaussian Noise**: 用于连续动作空间（DDPG/TD3）
- **Dummy Strategy**: 不添加探索噪声（用于 DDPG 的 OU Noise 内置探索）

### 回放缓冲区
- **ReplayBuffer**: 均匀采样的经验回放
- **PrioritizedReplayBuffer**: 基于 TD 误差的优先级回放

## 依赖要求

- Python ≥ 3.10
- PyTorch ≥ 2.9
- Gymnasium ≥ 1.2
- NumPy ≥ 2.2
- Matplotlib ≥ 3.10
- Draccus ≥ 0.11（配置管理）
- PyBullet（用于机器人环境，可选）

完整依赖列表见 `pyproject.toml`。

## 参考文献

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (DQN)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (DDQN)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) (DDPG)
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (TD3)
