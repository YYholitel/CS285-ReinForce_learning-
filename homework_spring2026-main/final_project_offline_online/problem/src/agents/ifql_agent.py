from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        num_samples: int = 32,
        expectile: float = 0.9,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        # TODO(student): Create flow actor

        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and value function

        # TODO(student): Create optimizers for all the above models

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        """
        # TODO(student): Implement the expectile loss
        return ...

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        # TODO(student): Implement the value function update
        
        # TODO(student): Update value function
        
        return ...

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        # TODO(student): Implement rejection sampling

        return ...

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        # TODO(student): Implement get action
        return ...

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        # TODO(student): Implement euler integration to get flow action
        
        return ...

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        # TODO(student): Implement Q-function update
        
        # TODO(student): Update Q-function
        
        return ...


    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        # TODO(student): Implement flow actor update
        
        # TODO(student): Update flow actor
        
        return ...


    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_value(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        pass 
