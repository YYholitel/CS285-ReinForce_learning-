from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch


class DrGRPO(RLAlgorithm):
    """DrGRPO removes the GRPO sequence-length normalization and std scaling."""

    name = "dr_grpo"

    def update(
        self,
        model: PolicyModel,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        del model, optimizer, rollout, grad_accum_steps
        # TODO(student): implement DrGRPO.
        # Start from your GRPO implementation, then make the two intended changes:
        #   1. use the DrGRPO advantage convention (configured in online/train_rm_grpo.py),
        #   2. remove the per-sequence length normalization inside the surrogate.
        raise NotImplementedError("Implement DrGRPO.update in the student starter.")
