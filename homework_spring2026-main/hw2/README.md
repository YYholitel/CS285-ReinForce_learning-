# CS285 Homework 2: Policy Gradients 与 DQN

本项目实现了 Policy Gradient（REINFORCE）及其多个改进版本，并额外从头实现了一个用于 `CartPole-v1` 的 DQN。

## 1. 实现内容

### 1.1 Policy Gradient 主作业

Policy Gradient 的完整训练流程位于 `src` 目录，包括：

- 使用 MLP 表示策略网络；
- 离散动作使用 `Categorical` 分布；
- 连续动作使用可学习均值和标准差的 `Normal` 分布；
- 根据当前策略采样动作并收集完整轨迹；
- 使用 REINFORCE loss 更新 Actor；
- 计算整条轨迹的 discounted return；
- 计算 Reward-to-Go；
- 对 advantage 进行标准化；
- 使用 Value Critic 作为 baseline；
- 使用 MSE loss 训练 Critic；
- 使用 Generalized Advantage Estimation（GAE）；
- 记录训练/评估回报、episode 长度、Actor loss 和 Baseline loss；
- 保存 CSV 日志、训练配置和模型 checkpoint；
- 通过 W&B 记录指标和 rollout 视频。

核心文件：

| 文件 | 功能 |
|---|---|
| `src/agents/pg_agent.py` | Return、Reward-to-Go、Advantage、Baseline 和 GAE |
| `src/networks/policies.py` | 策略网络、动作分布和 Actor 更新 |
| `src/networks/critics.py` | Value Critic 和 MSE 更新 |
| `src/infrastructure/utils.py` | 轨迹采集与指标计算 |
| `src/scripts/run.py` | 完整训练、评估、日志和模型保存流程 |
| `src/scripts/watch_cartpole.py` | 加载 checkpoint 并播放训练后的策略 |

### 1.2 DQN 扩展实现

`scratch_rl/dqn_cartpole` 中还实现了一套独立的 DQN，包括：

- 两层隐藏层的 Q Network；
- ε-greedy 探索策略；
- Replay Buffer 随机批量采样；
- Bellman TD target；
- Q-value MSE loss；
- 独立 Target Network；
- 定期同步 Target Network；
- ε 从 1.0 线性衰减到 0.05；
- 保存和加载训练后的 DQN 权重。

核心文件：

| 文件 | 功能 |
|---|---|
| `scratch_rl/dqn_cartpole/q_network.py` | Q 网络 |
| `scratch_rl/dqn_cartpole/replay_buffer.py` | 经验回放 |
| `scratch_rl/dqn_cartpole/dqn_agent.py` | 动作选择、Bellman target 和网络更新 |
| `scratch_rl/dqn_cartpole/train.py` | DQN 训练入口 |
| `scratch_rl/dqn_cartpole/watch.py` | DQN 动画演示 |

### 1.3 已完成的 CartPole 实验

现有训练日志中，各方法在 `CartPole-v0` 上取得过以下结果：

| 方法 | 最终 Eval Average Return | 最佳 Eval Average Return |
|---|---:|---:|
| REINFORCE | 97.4 | 155 |
| Reward-to-Go | 142.3 | 200 |
| RTG + Advantage Normalization | 106.75 | 200 |
| RTG + Baseline | 176 | 200 |
| RTG + Baseline + GAE | 200 | 200 |

其中 `RTG + Baseline + GAE` 的现有实验最终达到 `CartPole-v0` 的 200 回报上限。

### 1.4 当前尚未完成的部分

`scratch_rl/tabular_gridworld.py` 目前只包含 4×4 GridWorld 的状态、动作和坐标转换定义，还没有完整的环境转移或学习算法，因此不作为本 README 的可运行算法。

## 2. 进入项目目录

以下命令均在 PowerShell 中运行：

```powershell
cd J:\CS285\homework_spring2026-main\hw2
```

## 3. 安装环境

项目要求 Python 3.10 或更高版本，并使用 `uv` 管理依赖。

如果还没有安装 `uv`：

```powershell
winget install --id=astral-sh.uv -e
```

进入 `hw2` 后安装锁定的依赖：

```powershell
uv sync
```

验证环境：

```powershell
uv run python -c "import torch, gym; print('PyTorch:', torch.__version__); print('Gym:', gym.__version__)"
```

> 第一次执行 `uv sync` 需要联网下载依赖。Policy Gradient 使用项目中锁定的 `gym==0.25.2`，不要随意升级，否则 Gym API 可能不兼容。

## 4. 运行 Policy Gradient

训练入口是 `src/scripts/run.py`。训练结果会保存在：

```text
exp/<环境名>_<实验名>_sd<随机种子>_<时间>/
├── agent.pt       # 模型权重
├── flags.json     # 训练参数
├── log.csv        # 训练和评估指标
└── log.pkl        # 完整日志
```

### 4.1 原始 REINFORCE

每个时间步使用整条轨迹的 discounted return：

```powershell
uv run python src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name reinforce --no_gpu
```

### 4.2 Reward-to-Go

```powershell
uv run python src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name reinforce_rtg --use_reward_to_go --no_gpu
```

### 4.3 Reward-to-Go + Advantage Normalization

```powershell
uv run python src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name reinforce_rtg_na --use_reward_to_go --normalize_advantages --no_gpu
```

### 4.4 Reward-to-Go + Value Baseline

```powershell
uv run python src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name reinforce_baseline --use_reward_to_go --use_baseline --normalize_advantages --no_gpu
```

### 4.5 Reward-to-Go + Baseline + GAE

```powershell
uv run python src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name reinforce_gae --use_reward_to_go --use_baseline --gae_lambda 0.95 --normalize_advantages --no_gpu
```

### 4.6 快速冒烟测试

先用少量迭代确认程序能够正常运行：

```powershell
uv run python src/scripts/run.py --env_name CartPole-v0 -n 2 -b 500 -eb 200 --exp_name smoke_test --use_reward_to_go --no_gpu
```

### 4.7 Weights & Biases

训练脚本默认初始化 W&B。如果需要在线记录，先登录：

```powershell
uv run wandb login
```

如果只想在本地运行，可在当前 PowerShell 会话中切换为离线模式：

```powershell
$env:WANDB_MODE="offline"
```

恢复在线模式：

```powershell
Remove-Item Env:WANDB_MODE
```

## 5. 演示训练好的 Policy Gradient 模型

打开 `src/scripts/watch_cartpole.py`，把 `CHECKPOINT` 改成实际生成的 `agent.pt`，例如：

```python
CHECKPOINT = r"J:\CS285\homework_spring2026-main\hw2\exp\CartPole-v0_reinforce_rtg_sd1_YYYYMMDD_HHMMSS\agent.pt"
```

同时确保脚本中 `PGAgent` 的网络结构和训练参数与该 checkpoint 一致。然后运行：

```powershell
uv run python src/scripts/watch_cartpole.py
```

程序会打开 CartPole 动画窗口，并使用概率最大的动作进行演示。

## 6. 运行 DQN

DQN 位于 `scratch_rl/dqn_cartpole`。它使用 `gymnasium`，而不是主作业使用的旧版 `gym`。可以通过 `uv --with` 临时添加该依赖，无需修改项目依赖文件。

### 6.1 训练

```powershell
cd J:\CS285\homework_spring2026-main\hw2\scratch_rl\dqn_cartpole
uv run --with "gymnasium[classic-control]" python train.py
```

训练完成后，当前目录会生成：

```text
dqn_cartpole.pt
```

默认配置包括：

- 300 个 episode；
- 50,000 容量的 replay buffer；
- ε 从 1.0 线性衰减至 0.05；
- 每 500 个环境步同步一次 target network。

### 6.2 观看 DQN 智能体

保持当前目录为 `scratch_rl/dqn_cartpole`：

```powershell
uv run --with "gymnasium[classic-control]" python watch.py
```

脚本会读取当前目录的 `dqn_cartpole.pt` 并运行 10 个 episode。

## 7. 可视化及其位置

本项目有四类可视化/结果输出。

### 7.1 已生成的算法对比图

仓库中已有的完整对比图位于：

```text
scratch_rl/cartpole_method_comparison.png
```

另有一张部分实验对比图：

```text
scratch_rl/cartpole_method_comparison_partial.png
```

它们比较了 REINFORCE、Reward-to-Go、Advantage Normalization、Baseline、GAE 和 DQN 的学习趋势。

### 7.2 重新绘制算法对比图

回到 `hw2` 目录：

```powershell
cd J:\CS285\homework_spring2026-main\hw2
uv run python scratch_rl/plot_compare.py
```

完整绘图需要：

- `exp` 中存在各个 Policy Gradient 实验的 `log.csv` 和 `flags.json`；
- `scratch_rl/dqn_cartpole/dqn_returns.npy` 存在。

输出文件为：

```text
scratch_rl/cartpole_method_comparison.png
```

如果只有部分实验数据：

```powershell
uv run python scratch_rl/plot_compare.py --allow-missing
```

> 图中的 PG 横轴是 training iteration，DQN 横轴是 episode，因此该图适合比较学习趋势，不能作为严格的 sample-efficiency 对比。

### 7.3 训练曲线和数值日志

每次 Policy Gradient 训练的指标保存在：

```text
exp/<实验目录>/log.csv
```

模型和参数位于同一个实验目录：

```text
exp/<实验目录>/agent.pt
exp/<实验目录>/flags.json
exp/<实验目录>/log.pkl
```

可以用 Excel、Python、Matplotlib 或 W&B 打开 `log.csv` 查看 `Eval_AverageReturn`、`Train_AverageReturn`、loss 和 episode length。

### 7.4 W&B 在线可视化

在线模式运行时，训练指标会记录到 W&B 的 `cs285_hw2` project。终端会输出对应的 run URL；打开该链接即可查看交互式训练曲线。

如果使用 `$env:WANDB_MODE="offline"`，数据只保存在 W&B 提示的本地离线目录，不会自动上传。

### 7.5 CartPole 动画

- Policy Gradient 动画：运行 `src/scripts/watch_cartpole.py`；
- DQN 动画：在 `scratch_rl/dqn_cartpole` 中运行 `watch.py`。

两者都会打开独立的 CartPole 图形窗口，而不是生成静态图片。

## 8. 常用参数

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--env_name` | Gym 环境 | `CartPole-v0` |
| `-n`, `--n_iter` | 训练迭代数 | `200` |
| `-b`, `--batch_size` | 每轮至少采集的环境步数 | `1000` |
| `-eb`, `--eval_batch_size` | 每次评估采集的环境步数 | `400` |
| `--discount` | 折扣因子 γ | `1.0` |
| `-lr`, `--learning_rate` | Actor 学习率 | `5e-3` |
| `--use_reward_to_go` | 启用 Reward-to-Go | 关闭 |
| `--use_baseline` | 启用 Value Critic | 关闭 |
| `--gae_lambda` | 启用 GAE 并设置 λ | `None` |
| `--normalize_advantages` | 标准化 advantage | 关闭 |
| `--no_gpu` | 强制使用 CPU | 关闭 |
| `--seed` | 随机种子 | `1` |

查看完整参数：

```powershell
uv run python src/scripts/run.py --help
```

## 9. 常见问题

### 找不到 `uv`

安装后重新打开 PowerShell，再执行 `uv --version`。

### W&B 要求登录

执行 `uv run wandb login`，或者设置 `$env:WANDB_MODE="offline"`。

### DQN 报错 `No module named gymnasium`

使用文档中的命令：

```powershell
uv run --with "gymnasium[classic-control]" python train.py
```

### 演示脚本找不到模型

PG 演示需要修改 `watch_cartpole.py` 中的 `CHECKPOINT`；DQN 演示则要求当前目录中存在 `dqn_cartpole.pt`。

### 动画窗口没有出现

确保是在有桌面环境的本机运行，并已安装 Classic Control 的渲染依赖。远程无图形界面的服务器通常不能直接打开窗口。
