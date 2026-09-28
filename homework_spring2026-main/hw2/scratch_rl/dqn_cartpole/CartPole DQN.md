# CartPole DQN

This project implements a minimal Deep Q-Network (DQN) for `CartPole-v1`.

The goal is to understand value-based reinforcement learning by building the full training loop from scratch:

```text
QNetwork
→ epsilon-greedy exploration
→ replay buffer
→ Bellman target
→ MSE loss
→ target network
→ trained CartPole policy