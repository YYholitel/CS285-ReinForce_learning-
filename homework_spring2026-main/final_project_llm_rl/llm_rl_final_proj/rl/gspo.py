from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch


class GSPO(RLAlgorithm):
    """Sequence-level clipped surrogate using geometric-mean likelihood ratios."""

    name = "gspo"

    def update(
        self,
        model: PolicyModel,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        del model, optimizer, rollout, grad_accum_steps
        # TODO(student): implement GSPO.
        # The main change relative to GRPO is that you should aggregate token log-ratios into
        # one sequence-level ratio before applying PPO-style clipping.
        raise NotImplementedError("Implement GSPO.update in the student starter.")
