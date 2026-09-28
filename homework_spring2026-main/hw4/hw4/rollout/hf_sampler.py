from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers.generation import GenerationConfig

from hw4.models.load import tokenize_chat_prompts
from hw4.models.logprobs import build_completion_mask, compute_per_token_logprobs
from hw4.rollout.sampler_base import RolloutOutput, Sampler


@dataclass
class SamplingConfig:
    min_new_tokens: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    do_sample: bool = True


class HFSampler(Sampler):
    def __init__(self, tokenizer, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def rollout(
        self,
        policy_model: torch.nn.Module,
        prompt_messages: List[List[Dict[str, str]]],
        task_names: List[str],
        task_metas: List[Dict[str, Any]],
        group_size: int,
        sampling: SamplingConfig,
        max_prompt_tokens: Optional[int] = None,
        output_to_cpu: bool = False,
    ) -> RolloutOutput:
        assert len(prompt_messages) == len(task_names) == len(task_metas)
        B = len(prompt_messages)

        input_ids, attention_mask = tokenize_chat_prompts(
            self.tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            max_prompt_tokens=max_prompt_tokens,
            device=self.device,
        )
        prompt_input_len = int(input_ids.shape[1])
        pad_id = int(self.tokenizer.pad_token_id)

        do_sample = bool(sampling.do_sample and (sampling.temperature > 0.0))
        temperature = float(max(sampling.temperature, 1e-5))
        gen_cfg = GenerationConfig(
            min_new_tokens=max(0, min(int(sampling.min_new_tokens), int(sampling.max_new_tokens))),
            max_new_tokens=sampling.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=sampling.top_p,
            top_k=sampling.top_k if sampling.top_k > 0 else 0,
            repetition_penalty=sampling.repetition_penalty,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=group_size,
            num_beams=1,
            use_cache=True,
        )

        was_training = bool(policy_model.training)
        had_gc = bool(getattr(policy_model, "is_gradient_checkpointing", False))
        if had_gc and hasattr(policy_model, "gradient_checkpointing_disable"):
            policy_model.gradient_checkpointing_disable()
            policy_model.config.use_cache = True
        policy_model.eval()
        try:
            vocab_size = int(getattr(policy_model.config, "vocab_size", 0))
            if vocab_size > 0:
                mx = int(input_ids.max().item())
                mn = int(input_ids.min().item())
                if mn < 0 or mx >= vocab_size:
                    raise RuntimeError(f"Prompt token ids out of range: min={mn}, max={mx}, vocab={vocab_size}")

            sequences = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
                use_cache=True,
            )
            full_attention = (sequences != pad_id).long()
            completion_ids = sequences[:, prompt_input_len:]
            completion_texts: List[str] = []
            for row in completion_ids:
                if (row == pad_id).any():
                    n = int((row != pad_id).sum().item())
                    row = row[:n]
                completion_texts.append(self.tokenizer.decode(row, skip_special_tokens=True))

            old_logp = compute_per_token_logprobs(
                policy_model,
                sequences,
                full_attention,
                enable_grad=False,
            )

            # Reference logprobs are computed from the base model by disabling adapters.
            if not hasattr(policy_model, "disable_adapter"):
                raise RuntimeError("Policy model must support disable_adapter() for LoRA reference logprobs.")
            with policy_model.disable_adapter():
                ref_logp = compute_per_token_logprobs(
                    policy_model,
                    sequences,
                    full_attention,
                    enable_grad=False,
                )

            completion_mask = build_completion_mask(
                input_ids=sequences,
                attention_mask=full_attention,
                prompt_input_len=prompt_input_len,
                pad_token_id=pad_id,
            )
        finally:
            if had_gc and hasattr(policy_model, "gradient_checkpointing_enable"):
                policy_model.gradient_checkpointing_enable()
                if hasattr(policy_model, "enable_input_require_grads"):
                    policy_model.enable_input_require_grads()
                policy_model.config.use_cache = False
            if was_training:
                policy_model.train()

        prompt_messages_rep: List[List[Dict[str, str]]] = []
        task_names_rep: List[str] = []
        task_metas_rep: List[Dict[str, Any]] = []
        for i in range(B):
            for _ in range(group_size):
                prompt_messages_rep.append(prompt_messages[i])
                task_names_rep.append(task_names[i])
                task_metas_rep.append(task_metas[i])

        if output_to_cpu:
            sequences = sequences.cpu()
            full_attention = full_attention.cpu()
            completion_mask = completion_mask.cpu()
            old_logp = old_logp.cpu()
            ref_logp = ref_logp.cpu()

        return RolloutOutput(
            prompt_messages=prompt_messages_rep,
            completion_texts=completion_texts,
            input_ids=sequences,
            attention_mask=full_attention,
            completion_mask=completion_mask,
            old_logprobs=old_logp,
            ref_logprobs=ref_logp,
            prompt_input_len=prompt_input_len,
            group_size=group_size,
            task_names=task_names_rep,
            task_metas=task_metas_rep,
        )
