from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch


class GRPO(RLAlgorithm):
    """GRPO update with a PPO-style clipped surrogate over completion tokens."""

    name = "grpo"

    def update(
        self,
        model: PolicyModel,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        del model, optimizer, rollout, grad_accum_steps
        # TODO(student): implement one GRPO training iteration.
        # The intended structure is:
        #   1. loop over PPO epochs,
        #   2. iterate over rollout minibatches,
        #   3. recompute token log-probabilities under the current policy,
        #   4. form PPO ratios against mb.old_logprobs,
        #   5. apply token-level clipping with the sequence-level GRPO averaging used in this codebase,
        #   6. add KL regularization against mb.ref_logprobs,
        #   7. handle gradient accumulation / clipping / optimizer steps,
        #   8. return the logged metrics expected by the training script.
        raise NotImplementedError("Implement GRPO.update in the student starter.")
