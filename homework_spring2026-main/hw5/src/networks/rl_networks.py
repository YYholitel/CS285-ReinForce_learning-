from typing import Optional

from torch import nn

import torch
from torch import distributions

from infrastructure import pytorch_util as ptu
from infrastructure.distributions import make_tanh_transformed, make_multi_normal


class Policy(nn.Module):
    """
    Base policy, which can take an observation and output a distribution over actions.

    This class implements `forward()` which takes a (batched) observation and returns a distribution over actions.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                assert fixed_std is None
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=2*ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)
            else:
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)

                if self.fixed_std:
                    self.std = 1.0
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )


    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(obs), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.net(obs)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                return make_multi_normal(mean, std)

        return action_distribution


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = False,
    ):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=ob_dim,
            output_size=ac_dim,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)
        self.use_tanh = use_tanh

    def forward(self, obs):
        acs = self.net(obs)
        if self.use_tanh:
            acs = torch.tanh(acs)
        return acs


class VectorFieldPolicy(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
    ):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=ob_dim + ac_dim + 1,
            output_size=ac_dim,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

    def forward(self, obs, acs, times=None):
        if times is None:
            times = torch.zeros((*acs.shape[:-1], 1), device=acs.device)
        vs = self.net(torch.cat([obs, acs, times], dim=-1))
        return vs


class Value(nn.Module):
    def __init__(self, ob_dim, n_layers, size):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class EnsembleCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size, n_ensembles):
        super().__init__()
        self.net = ptu.build_ensemble_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
            n=n_ensembles,
        ).to(ptu.device)

    def forward(self, obs, acs):
        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)


class LogParam(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.log_param = nn.Parameter(torch.tensor(init_value, device=ptu.device).log())

    def forward(self):
        return self.log_param.exp()
