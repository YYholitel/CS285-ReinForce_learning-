from typing import Optional, Tuple

import gymnasium
import numpy as np
import ogbench
import torch
import torch.nn as nn

import infrastructure.pytorch_util as ptu
from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.utils import EpisodeMonitor
from networks.rl_networks import Policy, EnsembleCritic, VectorFieldPolicy



def dsrl_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 512,
    num_layers: int = 4,
    learning_rate: float = 3e-4,
    discount: float = 0.99,
    target_update_rate: float = 0.005,
    flow_steps: int = 10,
    noise_scale: float = 1.0,
    total_steps: int = 1000000,
    batch_size: int = 256,
    **kwargs,
):
    def make_bc_flow_actor(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        # TODO(student): Create BC flow actor - refer to FQL config
        return ...

    def make_noise_actor(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        # TODO(student): Create noise actor - this can be a regular MLP
        return ...

    def make_critic(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        # TODO(student): Create critic - will be a ensemble of Q-functions
        return ...
    
    def make_noise_critic(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        # TODO(student): Create noise critic - will be a ensemble of Q-functions
        return ...

    def make_optimizer(params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_env_and_dataset() -> Tuple[gymnasium.Env, ReplayBuffer]:
        env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        dataset = ReplayBuffer(capacity=len(train_dataset['observations']))
        dataset.size = len(train_dataset['observations'])
        dataset.observations = train_dataset['observations']
        dataset.next_observations = train_dataset['next_observations']
        dataset.actions = train_dataset['actions']
        dataset.rewards = train_dataset['rewards']
        dataset.dones = 1 - train_dataset['masks']

        return env, dataset

    log_string = f"{exp_name or 'dsrl'}_{env_name}"

    config = {
        "agent_kwargs": {
            "make_bc_flow_actor": make_bc_flow_actor,
            "make_bc_flow_actor_optimizer": make_optimizer,
            "make_noise_actor": make_noise_actor,
            "make_noise_actor_optimizer": make_optimizer,
            "make_critic": make_critic,
            "make_critic_optimizer": make_optimizer,

            "discount": discount,
            "target_update_rate": target_update_rate,
            "flow_steps": flow_steps,
            "noise_scale": noise_scale,
        },
        "agent": "dsrl",
        "log_name": log_string,
        "make_env_and_dataset": make_env_and_dataset,
        "total_steps": total_steps,
        "env_name": env_name,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        **kwargs,
    }

    return config
