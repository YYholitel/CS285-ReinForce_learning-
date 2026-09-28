from typing import Tuple, Optional

import gym

import numpy as np
import torch
import torch.nn as nn

from networks.policies import MLPPolicy
from networks.critics import StateActionCritic
from infrastructure import pytorch_util as ptu

from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.clip_action import ClipAction
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics


def sac_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 128,
    num_layers: int = 3,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    total_steps: int = 1000000,
    random_steps: int = 5000,
    training_starts: int = 10000,
    batch_size: int = 128,
    replay_buffer_capacity: int = 1000000,
    ep_len: Optional[int] = None,
    discount: float = 0.99,
    use_soft_target_update: bool = False,
    target_update_period: Optional[int] = None,
    soft_target_update_rate: Optional[float] = None,
    # Actor-critic configuration
    num_critic_updates: int = 1,
    # Settings for multiple critics
    num_critic_networks: int = 1,
    target_critic_backup_type: str = "mean",  # One of "doubleq", "min", or "mean"
    # Soft actor-critic
    backup_entropy: bool = True,
    use_entropy_bonus: bool = True,
    temperature: float = 0.1,
    actor_fixed_std: Optional[float] = None,
    use_tanh: bool = True,
    # Automatic temperature tuning (Section 3.6)
    auto_tune_temperature: bool = False,
    alpha_learning_rate: float = 3e-4,
):
    def make_critic(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        return StateActionCritic(
            ob_dim=np.prod(observation_shape),
            ac_dim=action_dim,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_actor(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        assert len(observation_shape) == 1
        if actor_fixed_std is not None:
            return MLPPolicy(
                ac_dim=action_dim,
                ob_dim=np.prod(observation_shape),
                discrete=False,
                n_layers=num_layers,
                layer_size=hidden_size,
                use_tanh=use_tanh,
                state_dependent_std=False,
                fixed_std=actor_fixed_std,
            )
        else:
            return MLPPolicy(
                ac_dim=action_dim,
                ob_dim=np.prod(observation_shape),
                discrete=False,
                n_layers=num_layers,
                layer_size=hidden_size,
                use_tanh=use_tanh,
                state_dependent_std=True,
            )

    def make_actor_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=actor_learning_rate)

    def make_critic_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=critic_learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env(eval=False, render=False):
        return RecordEpisodeStatistics(
            ClipAction(
                RescaleAction(
                    gym.make(
                        env_name, render_mode="rgb_array" if render else None
                    ),
                    -1,
                    1,
                )
            )
        )

    log_string = "{}_{}".format(
        env_name,
        exp_name or "sac",
    )

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_critic_optimizer": make_critic_optimizer,
            "make_critic_schedule": make_lr_schedule,
            "make_actor": make_actor,
            "make_actor_optimizer": make_actor_optimizer,
            "make_actor_schedule": make_lr_schedule,
            "num_critic_updates": num_critic_updates,
            "discount": discount,
            "num_critic_networks": num_critic_networks,
            "target_critic_backup_type": target_critic_backup_type,
            "use_entropy_bonus": use_entropy_bonus,
            "backup_entropy": backup_entropy,
            "temperature": temperature,
            "auto_tune_temperature": auto_tune_temperature,
            "alpha_learning_rate": alpha_learning_rate,
            "target_update_period": target_update_period
            if not use_soft_target_update
            else None,
            "soft_target_update_rate": soft_target_update_rate
            if use_soft_target_update
            else None,
        },
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "total_steps": total_steps,
        "random_steps": random_steps,
        "training_starts": training_starts,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "make_env": make_env,
    }


# Config registry
configs = {
    "sac": sac_config,
}
