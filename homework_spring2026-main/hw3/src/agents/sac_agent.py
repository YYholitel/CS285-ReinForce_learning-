from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

from infrastructure import pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "min" or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
        # Automatic temperature tuning (Section 3.5)
        auto_tune_temperature: bool = False,
        alpha_learning_rate: float = 3e-4,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "min",
            "mean",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        # Automatic temperature tuning (Section 3.5)
        self.auto_tune_temperature = auto_tune_temperature
        if self.auto_tune_temperature:
            # TODO(Section 3.5): Initialize log_alpha, alpha_optimizer, and target_entropy
            # Hint: Initialize log_alpha to log(temperature) so alpha starts at the given temperature
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None
            # ENDTODO

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_temperature(self) -> float:
        """
        Get the current temperature value (either fixed or learned).
        """
        if self.auto_tune_temperature:
            # TODO(Section 3.5): Return the current learned temperature
            # skip here until we implement the temperature tuning
            return None
            # ENDTODO
        else:
            return self.temperature

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "mean", we average the Q-values across critics (single-Q baseline).
         - for "min", we take the minimum of the two critics' predictions (clipped double-Q).

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        # TODO(Section 3.6): Implement the "min" backup strategy (clipped double-Q).
        if self.target_critic_backup_type == "mean":
            next_qs = next_qs.mean(dim=0)
        elif self.target_critic_backup_type == "min":
            next_qs = None
        else:
            raise ValueError(
                f"Invalid critic backup strategy {self.target_critic_backup_type}"
            )
        # ENDTODO

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(Section 3.2): Sample from the actor and compute next Q-values
            next_action_distribution = None
            next_action = None
            next_qs = None
            # ENDTODO

            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(Section 3.3): Add entropy bonus to the target values for SAC
                next_action_entropy = None
                # Hint: next_qs = ...
                # ENDTODO

            # Handle Q-values from multiple different target critic networks (if necessary)
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            # TODO(Section 3.2): Compute the target Q-value
            target_values = None
            # ENDTODO
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size,
            ), target_values.shape

        # TODO(Section 3.2): Update the critic
        # Predict Q-values
        q_values = None
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        loss = None
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """

        # TODO(Section 3.3): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        return None
        # ENDTODO

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # TODO(Section 3.4): Sample actions using reparameterization (replace the placeholder below)
        # Note: Think about whether to use .rsample() or .sample() here, and why...
        action = torch.zeros(batch_size, self.action_dim, device=obs.device) # replace this with the correct action
        assert action.shape == (batch_size, self.action_dim), action.shape
        # ENDTODO

        # TODO(Section 3.4): Compute Q-values for the sampled state-action pair (replace the placeholder below)
        q_values = torch.zeros(self.num_critic_networks, batch_size, device=obs.device) # replace this with the correct q_values
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape
        # ENDTODO

        # Compute log probabilities for alpha update (Section 3.5)
        log_prob = action_distribution.log_prob(action)

        # TODO(Section 3.4): Compute the actor loss (replace the placeholder below)
        loss = torch.tensor(0.0, device=obs.device) # replace this with the correct loss
        # ENDTODO

        return loss, torch.mean(self.entropy(action_distribution)), log_prob

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using reparameterization.
        """
        loss, entropy, log_prob = self.actor_loss_reparametrize(obs)

        # TODO(Section 3.3): Add the entropy bonus to the actor loss: loss -= [your entropy bonus here]
        pass
        # ENDTODO

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": loss.item(),
            "entropy": entropy.item(),
            "log_prob": log_prob.detach(),  # Detach for alpha update
        }

    def update_alpha(self, log_prob: torch.Tensor):
        """
        Update temperature parameter using dual gradient descent (Section 3.5).

        The goal is to maintain a minimum entropy level (target_entropy).
        If current entropy is below target, alpha should increase to encourage exploration.
        If current entropy is above target, alpha should decrease.

        Args:
            log_prob: Log probability of actions from current policy, shape (batch_size,)
                     Should be detached from actor gradients.

        Returns:
            dict: Dictionary with alpha and alpha_loss for logging
        """
        if not self.auto_tune_temperature:
            return {}

        # TODO(Section 3.5): Implement dual gradient descent for temperature tuning
        alpha = None
        alpha_loss = None

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "alpha": alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }
        # ENDTODO

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []
        # TODO(Section 3.2): Update the critic for num_critic_updates steps
        for _ in range(self.num_critic_updates):
            info = None
            critic_infos.append(info)
        # ENDTODO

        # TODO(Section 3.3): Enable the actor update (once you have implemented entropy)
        actor_info = {}
        # ENDTODO

        # Update alpha (temperature) using dual gradient descent (Section 3.5)
        if self.auto_tune_temperature:
            alpha_info = self.update_alpha(actor_info["log_prob"])
        else:
            alpha_info = {}

        # TODO(Section 3.2): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        pass
        # ENDTODO

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        # Remove log_prob from actor_info before returning (only needed for alpha update)
        if "log_prob" in actor_info:
            del actor_info["log_prob"]

        result = {
            **actor_info,
            **critic_info,
            **alpha_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }

        # Always log temperature (whether fixed or learned)
        result["temperature"] = self.get_temperature()

        return result
