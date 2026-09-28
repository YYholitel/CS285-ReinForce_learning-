from __future__ import annotations

import os
from typing import Any, Dict

import torch


def resolve_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return device, dtype


def require_cuda_if_requested() -> None:
    if os.environ.get("REQUIRE_CUDA") != "1":
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "This run was launched from a GPU-backed entrypoint with REQUIRE_CUDA=1, "
        "but torch.cuda.is_available() is False. Refusing to continue on CPU."
    )


def get_hardware_metrics(device: torch.device) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "setup/device_type": device.type,
        "setup/torch_cuda_is_available": bool(torch.cuda.is_available()),
        "setup/cuda_device_count": float(torch.cuda.device_count()),
    }
    if device.type != "cuda":
        return metrics

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    bf16_supported = False
    try:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16_supported = False
    metrics.update(
        {
            "setup/cuda_current_device_index": float(idx),
            "setup/cuda_device_name": props.name,
            "setup/cuda_total_memory_mb": float(props.total_memory / (1024**2)),
            "setup/cuda_multiprocessor_count": float(props.multi_processor_count),
            "setup/cuda_compute_capability_major": float(props.major),
            "setup/cuda_compute_capability_minor": float(props.minor),
            "setup/cuda_bf16_supported": bf16_supported,
            **get_cuda_memory_metrics(prefix="setup"),
        }
    )
    return metrics


def get_model_device_metrics(model: torch.nn.Module) -> Dict[str, Any]:
    devices = sorted({str(p.device) for p in model.parameters()})
    out: Dict[str, Any] = {
        "setup/model_param_device_count": float(len(devices)),
        "setup/model_param_devices": ",".join(devices),
    }
    if devices:
        out["setup/model_first_param_device"] = devices[0]
    return out


def get_cuda_memory_metrics(prefix: str) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        f"{prefix}/cuda_memory_allocated_mb": float(torch.cuda.memory_allocated() / (1024**2)),
        f"{prefix}/cuda_memory_reserved_mb": float(torch.cuda.memory_reserved() / (1024**2)),
        f"{prefix}/cuda_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
        f"{prefix}/cuda_max_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
    }
