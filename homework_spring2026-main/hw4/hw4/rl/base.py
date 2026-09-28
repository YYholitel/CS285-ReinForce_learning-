from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from hw4.rollout.rollout_buffer import RolloutBatch


@dataclass
class AlgoConfig:
    ppo_epochs: int = 1
    minibatch_size: int = 8
    clip_eps: float = 0.1
    kl_coef: float = 0.02
    max_grad_norm: float = 0.5
    adv_clip: float = 5.0

    # Helpful for deterministic shuffling across runs.
    seed: int = 0


class RLAlgorithm:
    name: str = "base"

    def __init__(self, cfg: AlgoConfig):
        self.cfg = cfg
        self._num_updates = 0

    def _next_update_seed(self) -> int:
        seed = int(self.cfg.seed + self._num_updates)
        self._num_updates += 1
        return seed

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        raise NotImplementedError
