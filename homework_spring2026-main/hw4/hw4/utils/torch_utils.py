from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch


@dataclass
class RunningMeanStd:
    mean: float = 0.0
    var: float = 1.0
    count: float = 1e-4

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()
        batch_count = x.numel()

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count

        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x - self.mean) / (self.var ** 0.5 + eps)


def clip_grad_norm_(params, max_norm: float) -> float:
    if max_norm <= 0:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
