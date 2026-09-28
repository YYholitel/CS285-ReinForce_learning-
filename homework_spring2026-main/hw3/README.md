# CS285 HW3: Q-Learning and Actor-Critic Algorithms

This assignment covers Deep Q-Networks (DQN) for discrete action spaces and Soft Actor-Critic (SAC) for continuous action spaces.

## Prerequisites

This assignment requires **Python 3.10** and the `swig` system package (needed by `gym[box2d]`).

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install swig
```

**macOS (Homebrew):**
```bash
brew install swig
```

If you are unable to install `swig` or run into other environment issues, you can run all experiments on cloud GPUs using Modal (see the Modal section below).

## Setup

Install dependencies using uv:

```bash
uv sync
```

## Running Experiments

### DQN

```bash
# CartPole (debug)
uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml

# LunarLander (Double-DQN)
uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1

# MsPacman
uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml
```

### SAC

```bash
# InvertedPendulum (sanity check)
uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml

# HalfCheetah
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml

# Hopper with different Q-backup strategies
uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_clipq.yaml
```

### Modal (Cloud Compute)

First, authenticate with Modal:
```bash
uv run modal token new
```

Then launch experiments remotely:
```bash
# DQN on Modal
uv run modal run src/scripts/modal_run_dqn.py -- -cfg experiments/dqn/mspacman.yaml

# SAC on Modal
uv run modal run src/scripts/modal_run_sac.py -- -cfg experiments/sac/halfcheetah.yaml
```

Download results from the persistent volume:
```bash
uv run modal volume get hw3-ql-volume exp/
```

## Directory Structure

```
src/
├── agents/           # DQN and SAC agent implementations
├── networks/         # Neural network modules (policies, critics)
├── infrastructure/   # Utilities (replay buffer, logging, etc.)
├── configs/          # Configuration factories
└── scripts/          # Training scripts
experiments/          # YAML configuration files
```
