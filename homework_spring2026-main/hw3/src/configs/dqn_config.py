from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.frame_stack import FrameStack

import numpy as np
import torch
import torch.nn as nn

from configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
from infrastructure import pytorch_util as ptu
from networks.critics import DQNCritic


def basic_dqn_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    total_steps: int = 1000000,
    discount: float = 0.99,
    target_update_period: int = 1000,
    clip_grad_norm: Optional[float] = None,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 128,
    **kwargs
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        return DQNCritic(
            observation_shape=observation_shape,
            num_actions=num_actions,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (total_steps * 0.3, 0.1),  # Decay to 0.1 over 30% of steps (more gradual)
            (total_steps * 0.6, 0.02),  # Then decay to 0.02 by 60% of steps
        ],
        outside_value=0.02,
    )

    def make_env(eval=False, render=False):
        return RecordEpisodeStatistics(
            gym.make(env_name, render_mode="rgb_array" if render else None)
        )

    log_string = "{}_{}".format(
        env_name,
        exp_name or "dqn",
    )

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "exploration_schedule": exploration_schedule,
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }


class PreprocessAtari(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in [3, 4], f"Bad observation shape: {x.shape}"
        assert x.shape[-3:] == (4, 84, 84), f"Bad observation shape: {x.shape}"
        assert x.dtype == torch.uint8

        return x / 255.0


def atari_dqn_config(
    env_name: str,
    exp_name: Optional[str] = None,
    learning_rate: float = 1e-4,
    adam_eps: float = 1e-4,
    total_steps: int = 1000000,
    discount: float = 0.99,
    target_update_period: int = 2000,
    clip_grad_norm: Optional[float] = 10.0,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 32,
    **kwargs,
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        assert observation_shape == (
            4,
            84,
            84,
        ), f"Observation shape: {observation_shape}"

        return nn.Sequential(
            PreprocessAtari(),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, num_actions),
        ).to(ptu.device)

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate, eps=adam_eps)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            PiecewiseSchedule(
                [
                    (0, 1),
                    (20000, 1),
                    (total_steps / 2, 5e-1),
                ],
                outside_value=5e-1,
            ).value,
        )

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (20000, 1),
            (total_steps / 2, 0.01),
        ],
        outside_value=0.01,
    )

    # Import here to avoid circular imports and to make atari wrappers optional
    try:
        from infrastructure.atari_wrappers import wrap_deepmind
    except ImportError:
        raise ImportError("Please install atari dependencies: pip install gym[atari,accept-rom-license]")

    def make_env(eval=False, render=False):
        return wrap_deepmind(
            gym.make(env_name, render_mode="rgb_array" if render else None)
        )

    # Shorten env_name if it's too long (e.g., MsPacmanNoFrameskip-v0 -> MsPacman)
    short_env_name = env_name.replace("NoFrameskip-v0", "").replace("NoFrameskip-v4", "")

    log_string = "{}_{}".format(
        short_env_name,
        exp_name or "dqn",
    )

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "log_name": log_string,
        "exploration_schedule": exploration_schedule,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }


# Config registry
configs = {
    "dqn_basic": basic_dqn_config,
    "dqn_atari": atari_dqn_config,
}
