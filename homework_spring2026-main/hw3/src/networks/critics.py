import torch
from torch import nn
import numpy as np

from infrastructure import pytorch_util as ptu


class DQNCritic(nn.Module):
    """Critic network for DQN. Maps observations to Q-values for each action."""

    def __init__(self, observation_shape, num_actions, n_layers, size):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=int(np.prod(observation_shape)),
            output_size=num_actions,
            n_layers=n_layers,
            size=size,
        )

    def forward(self, obs):
        """
        Return Q-values for all actions.

        Args:
            obs: (batch_size, *observation_shape) observations
        Returns:
            qa_values: (batch_size, num_actions) Q-values for each action
        """
        # Flatten observations if needed
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return self.net(obs)


class StateActionCritic(nn.Module):
    """Critic network for SAC. Maps (state, action) pairs to Q-values."""

    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )

    def forward(self, obs, acs):
        """
        Return Q-value for the given state-action pair.

        Args:
            obs: (batch_size, ob_dim) observations
            acs: (batch_size, ac_dim) actions
        Returns:
            q_values: (batch_size,) Q-values
        """
        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)
