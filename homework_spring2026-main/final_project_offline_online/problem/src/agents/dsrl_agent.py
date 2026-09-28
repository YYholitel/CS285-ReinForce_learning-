from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_z_critic,
        make_z_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        noise_scale: float = 1.0,

        online_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.target_entropy = -action_dim

        # TODO(student): Create BC flow actor and target BC flow actor

        # TODO(student): Create noise policy

        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and z critic (for noise policy)

        # TODO(student): Create learnable entropy coefficient

        # TODO(student): Create optimizers for all the above models

        self.to(ptu.device)

    @property
    def alpha(self):
        # TODO(student): Allow access to the learnable entropy coefficient (tip: if you are learning log alpha, as in HW3, then when we want to use alpha, you should return the exponential of the log alpha)
        return ...

    @torch.compiler.disable
    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of BC flow from t=0 to t=1."""
        # TODO(student): Implement Euler integration of BC flow. Keep in mind that the target BC flow actor should be used
        # Also note that we can control what we use as the noise input (could be sampled from a noise policy or from a normal distribution)
        return ...

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        # TODO(student): Sample noise from the noise policy and use to sample actions from the BC flow policy
        return ...
    
    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        # TODO(student): Implement get action
        
        return ...

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic"""
        # TODO(student): Implement critic loss
        loss = ...
        
        # TODO(student): Update critic
        
        return ...
    
    def update_qz(self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        noises: torch.Tensor,
    ) -> dict:
        """Update z_critic."""
        
        # TODO(student): Implement z_critic loss
        loss = ...

        # TODO(student): Update z_critic
        
        return ...

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor"""
        # TODO(student): Implement BC flow loss
        loss = ...
        
        # TODO(student): Update BC flow actor
        
        return ...
    
    def update_noise_actor(self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor."""
        # TODO(student): Implement noise actor loss
        loss = ...
        
        # TODO(student): Update noise actor
        
        return ...

    def update_alpha(self) -> dict:
        """Update alpha."""
        # TODO(student): Implement alpha loss
        loss = ...
        
        # TODO(student): Update alpha
        
        return ...

    def update_target_critic(self) -> None:
        # TODO(student): Implement target critic update
        return ...

    def update_target_bc_flow_actor(self) -> None:
        # TODO(student): Implement target BC flow actor update
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
        # TODO(student): Update critic, z_critic, actor, noise actor, and alpha - feel free to modify this code according to your setup!
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_qz = self.update_qz(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics_noise_actor = self.update_noise_actor(observations)
        metrics_alpha = self.update_alpha()
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"noise_actor/{k}": v.item() for k, v in metrics_noise_actor.items()},
            **{f"alpha/{k}": v.item() for k, v in metrics_alpha.items()},
        }

        self.update_target_critic()
        self.update_target_bc_flow_actor()

        return metrics

