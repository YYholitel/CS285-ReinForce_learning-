from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class LoadedPolicyModel:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    trainable_params: int
    total_params: int
    lora_target_modules: List[str]


@dataclass
class LoadedInferenceModel:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase


def _build_model_kwargs(dtype: torch.dtype) -> Dict[str, Any]:
    return {"dtype": dtype}


def _prepare_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _normalize_targets(target_modules: Sequence[str]) -> List[str]:
    out = []
    for t in target_modules:
        t2 = t.strip()
        if t2:
            out.append(t2)
    if not out:
        raise ValueError("No LoRA target modules provided.")
    return out


def _filter_existing_target_suffixes(model: torch.nn.Module, suffixes: Sequence[str]) -> List[str]:
    linear_names = [
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    ]
    keep: List[str] = []
    for suffix in suffixes:
        if any(name.endswith(suffix) for name in linear_names):
            keep.append(suffix)
    if not keep:
        raise ValueError(
            "None of the requested LoRA target modules matched model Linear layers. "
            f"Requested={list(suffixes)[:16]}"
        )
    return sorted(set(keep))


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _ensure_input_require_grads(model: torch.nn.Module) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    if not hasattr(model, "get_input_embeddings"):
        return
    emb = model.get_input_embeddings()
    if emb is None:
        return
    if getattr(model, "_input_require_grads_hook", None) is not None:
        return

    def _set_requires_grad(_module, _inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)

    model._input_require_grads_hook = emb.register_forward_hook(_set_requires_grad)


def load_lora_policy_model_and_tokenizer(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    grad_checkpointing: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    lora_bias: str = "none",
) -> LoadedPolicyModel:
    tokenizer = _prepare_tokenizer(model_name)

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_build_model_kwargs(dtype=dtype),
    )
    if grad_checkpointing:
        base.gradient_checkpointing_enable()
        _ensure_input_require_grads(base)
        base.config.use_cache = False

    normalized_targets = _normalize_targets(lora_target_modules)
    matched_targets = _filter_existing_target_suffixes(base, normalized_targets)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=matched_targets,
        bias=lora_bias,
    )
    model = get_peft_model(base, lora_cfg)
    model.to(device)

    if grad_checkpointing:
        # Important for LoRA+checkpointing: this must be set on the wrapped model.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        _ensure_input_require_grads(model)
        if hasattr(model, "base_model"):
            _ensure_input_require_grads(model.base_model)
        model.config.use_cache = False

    # Always keep frozen base + trainable adapter discipline.
    for name, p in model.named_parameters():
        is_lora = "lora_" in name
        p.requires_grad_(is_lora)
        if is_lora and p.dtype != torch.float32:
            # Keep trainable adapter params in fp32 for optimizer stability.
            p.data = p.data.float()

    trainable_params, total_params = _count_params(model)
    return LoadedPolicyModel(
        model=model,
        tokenizer=tokenizer,
        trainable_params=trainable_params,
        total_params=total_params,
        lora_target_modules=matched_targets,
    )


def load_inference_model_and_tokenizer(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
) -> LoadedInferenceModel:
    tokenizer = _prepare_tokenizer(model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        **_build_model_kwargs(dtype=dtype),
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    else:
        model = base
    model.to(device)
    model.eval()
    return LoadedInferenceModel(model=model, tokenizer=tokenizer)


def resolve_adapter_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.is_file():
        raise ValueError(
            "Adapter path must be a directory produced by model.save_pretrained(...)"
        )
    return str(p)


def tokenize_chat_prompts(
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    add_generation_prompt: bool = True,
    max_prompt_tokens: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encs: List[torch.Tensor] = []
    for messages in messages_list:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )[0]
        if max_prompt_tokens is not None and ids.numel() > max_prompt_tokens:
            ids = ids[-max_prompt_tokens:]
        encs.append(ids)

    max_len = max(x.numel() for x in encs)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(encs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encs), max_len), dtype=torch.long)

    for i, ids in enumerate(encs):
        n = ids.numel()
        input_ids[i, max_len - n :] = ids
        attention_mask[i, max_len - n :] = 1

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask
