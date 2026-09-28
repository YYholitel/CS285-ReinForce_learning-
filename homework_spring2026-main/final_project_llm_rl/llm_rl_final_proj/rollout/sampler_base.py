from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class RolloutOutput:
    # Flattened over (batch_size * group_size)
    prompt_messages: List[List[Dict[str, str]]]
    completion_texts: List[str]

    # Model-space tensors for PPO-style updates
    input_ids: torch.Tensor       # [N, L]
    attention_mask: torch.Tensor  # [N, L]
    completion_mask: torch.Tensor # [N, L-1] float {0,1}
    old_logprobs: torch.Tensor    # [N, L-1]
    ref_logprobs: torch.Tensor    # [N, L-1]

    # Metadata
    prompt_input_len: int         # padded prompt length used for generate
    group_size: int
    task_names: List[str]         # [N] name per completion
    task_metas: List[Dict[str, Any]]  # [N] meta per completion (includes ground truth etc.)


class Sampler:
    def rollout(self, *args, **kwargs) -> RolloutOutput:
        raise NotImplementedError
