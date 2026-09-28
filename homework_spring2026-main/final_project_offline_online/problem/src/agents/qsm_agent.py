from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List

class QSMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float,
        flow_steps: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        # TODO(student): Create actor
        
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions)
        
        # TODO(student): Create optimizers for all the above models
        
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps

        betas = self.cosine_beta_schedule(flow_steps)
        self.register_buffer("betas", ...) # TODO(student): Implement betas
        self.register_buffer("alphas", ...) # TODO(student): Implement alphas
        self.register_buffer("alpha_hats", ...) # TODO(student): Implement alpha_hats

        self.to(ptu.device)
    
    def cosine_beta_schedule(self, timesteps):
        """
        Cosine annealing beta schedule
        """
        # TODO(student): Implement cosine annealing beta schedule
        return ...
    
    @torch.compiler.disable
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor):
        """
        DDPM sampling
        """
        # TODO(student): Implement DDPM sampling
        return ...
    
    def get_action(self, observation: torch.Tensor):
        """
        Used for evaluation.
        """
        # TODO(student): Implement get_action
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
        Update Critic
        """
        # TODO(student): Implement critic update
        
        # TODO(student): Update critic
        
        return ...
        
    @torch.compiler.disable
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor
        """

        # TODO(student): Implement actor update
        
        # TODO(student): Update actor
        
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        pass