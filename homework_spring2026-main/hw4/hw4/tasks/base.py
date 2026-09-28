from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TaskExample:
    # Task-defined metadata (ground truth, numbers, etc.)
    meta: Dict[str, Any]

    # Chat messages for the model
    messages: List[Dict[str, str]]

    # Human-readable name
    task_name: str


class Task:
    name: str = "base"

    def sample_train_batch(self, batch_size: int) -> List[TaskExample]:
        raise NotImplementedError

    def reward(self, example: TaskExample, completion_text: str) -> Tuple[float, Dict[str, Any]]:
        """Returns (reward, info)."""
        raise NotImplementedError

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Return dict of metrics."""
        raise NotImplementedError
