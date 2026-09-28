from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from llm_rl_final_proj.data.ultrafeedback import PreferenceExample


@dataclass
class PreferenceBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_response_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_response_mask: torch.Tensor
    prompt_texts: List[str]
    chosen_texts: List[str]
    rejected_texts: List[str]
    row_ids: List[str]
    avg_confidence: torch.Tensor
    avg_preference_strength: torch.Tensor
    avg_training_quality: torch.Tensor

    def to(self, device: torch.device) -> "PreferenceBatch":
        return PreferenceBatch(
            chosen_input_ids=self.chosen_input_ids.to(device, non_blocking=True),
            chosen_attention_mask=self.chosen_attention_mask.to(device, non_blocking=True),
            chosen_response_mask=self.chosen_response_mask.to(device, non_blocking=True),
            rejected_input_ids=self.rejected_input_ids.to(device, non_blocking=True),
            rejected_attention_mask=self.rejected_attention_mask.to(device, non_blocking=True),
            rejected_response_mask=self.rejected_response_mask.to(device, non_blocking=True),
            prompt_texts=self.prompt_texts,
            chosen_texts=self.chosen_texts,
            rejected_texts=self.rejected_texts,
            row_ids=self.row_ids,
            avg_confidence=self.avg_confidence.to(device, non_blocking=True),
            avg_preference_strength=self.avg_preference_strength.to(device, non_blocking=True),
            avg_training_quality=self.avg_training_quality.to(device, non_blocking=True),
        )


class PreferenceCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_prompt_tokens: int,
        max_response_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_response_tokens = int(max_response_tokens)
        if self.max_prompt_tokens <= 0:
            raise ValueError(f"max_prompt_tokens must be >= 1, got {self.max_prompt_tokens}")
        if self.max_response_tokens <= 0:
            raise ValueError(f"max_response_tokens must be >= 1, got {self.max_response_tokens}")

    def __call__(self, examples: Sequence[PreferenceExample]) -> PreferenceBatch:
        chosen_ids: List[torch.Tensor] = []
        rejected_ids: List[torch.Tensor] = []
        chosen_response_lengths: List[int] = []
        rejected_response_lengths: List[int] = []
        prompt_texts: List[str] = []
        chosen_texts: List[str] = []
        rejected_texts: List[str] = []
        row_ids: List[str] = []
        avg_confidence: List[float] = []
        avg_preference_strength: List[float] = []
        avg_training_quality: List[float] = []

        for ex in examples:
            chosen_full_ids, chosen_response_len = _tokenize_prompt_with_response(
                tokenizer=self.tokenizer,
                prompt_messages=ex.prompt_messages,
                response_text=ex.chosen_text,
                max_prompt_tokens=self.max_prompt_tokens,
                max_response_tokens=self.max_response_tokens,
            )
            rejected_full_ids, rejected_response_len = _tokenize_prompt_with_response(
                tokenizer=self.tokenizer,
                prompt_messages=ex.prompt_messages,
                response_text=ex.rejected_text,
                max_prompt_tokens=self.max_prompt_tokens,
                max_response_tokens=self.max_response_tokens,
            )
            chosen_ids.append(chosen_full_ids)
            rejected_ids.append(rejected_full_ids)
            chosen_response_lengths.append(chosen_response_len)
            rejected_response_lengths.append(rejected_response_len)
            prompt_texts.append(ex.prompt_text)
            chosen_texts.append(ex.chosen_text)
            rejected_texts.append(ex.rejected_text)
            row_ids.append(ex.row_id)
            avg_confidence.append(float(ex.avg_confidence if ex.avg_confidence is not None else 1.0))
            avg_preference_strength.append(
                float(ex.avg_preference_strength if ex.avg_preference_strength is not None else 5.0)
            )
            avg_training_quality.append(
                float(ex.avg_training_quality if ex.avg_training_quality is not None else 5.0)
            )

        max_seq_len = max(
            max(int(ids.numel()) for ids in chosen_ids),
            max(int(ids.numel()) for ids in rejected_ids),
        )
        chosen_input_ids, chosen_attention_mask, chosen_response_mask = _left_pad_sequences(
            chosen_ids,
            response_lengths=chosen_response_lengths,
            pad_token_id=int(self.tokenizer.pad_token_id),
            max_len=max_seq_len,
        )
        rejected_input_ids, rejected_attention_mask, rejected_response_mask = _left_pad_sequences(
            rejected_ids,
            response_lengths=rejected_response_lengths,
            pad_token_id=int(self.tokenizer.pad_token_id),
            max_len=max_seq_len,
        )
        return PreferenceBatch(
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            chosen_response_mask=chosen_response_mask,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=rejected_attention_mask,
            rejected_response_mask=rejected_response_mask,
            prompt_texts=prompt_texts,
            chosen_texts=chosen_texts,
            rejected_texts=rejected_texts,
            row_ids=row_ids,
            avg_confidence=torch.tensor(avg_confidence, dtype=torch.float32),
            avg_preference_strength=torch.tensor(avg_preference_strength, dtype=torch.float32),
            avg_training_quality=torch.tensor(avg_training_quality, dtype=torch.float32),
        )


def _tokenize_prompt_with_response(
    tokenizer: PreTrainedTokenizerBase,
    prompt_messages: Sequence[Dict[str, str]],
    response_text: str,
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> tuple[torch.Tensor, int]:
    prompt_ids = tokenizer.apply_chat_template(
        list(prompt_messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0]
    full_messages = list(prompt_messages) + [{"role": "assistant", "content": response_text}]
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )[0]

    if prompt_ids.numel() > max_prompt_tokens:
        drop = int(prompt_ids.numel() - max_prompt_tokens)
        prompt_ids = prompt_ids[drop:]
        full_ids = full_ids[drop:]

    response_len = int(full_ids.numel() - prompt_ids.numel())
    if response_len <= 0:
        raise ValueError("Response tokenization produced zero completion tokens; cannot score preference pair.")

    if response_len > max_response_tokens:
        full_ids = full_ids[: int(prompt_ids.numel() + max_response_tokens)]
        response_len = max_response_tokens

    return full_ids, int(response_len)


def _left_pad_sequences(
    ids_list: Sequence[torch.Tensor],
    *,
    response_lengths: Sequence[int],
    pad_token_id: int,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_len is None:
        max_len = max(int(ids.numel()) for ids in ids_list)
    batch_size = len(ids_list)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.float32)

    for i, (ids, response_len) in enumerate(zip(ids_list, response_lengths)):
        n = int(ids.numel())
        input_ids[i, max_len - n :] = ids
        attention_mask[i, max_len - n :] = 1
        if response_len > 0:
            response_mask[i, (max_len - 1) - response_len :] = 1.0
    return input_ids, attention_mask, response_mask
