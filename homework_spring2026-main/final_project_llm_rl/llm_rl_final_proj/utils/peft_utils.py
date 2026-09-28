from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


@contextmanager
def disable_adapter_if_possible(model: torch.nn.Module) -> Iterator[None]:
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return
    if hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers"):
        model.disable_adapter_layers()
        try:
            yield
        finally:
            model.enable_adapter_layers()
        return
    with nullcontext():
        yield
