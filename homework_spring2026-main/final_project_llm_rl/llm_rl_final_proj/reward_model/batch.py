from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from llm_rl_final_proj.data.ultrafeedback import GenerationExample, PreferenceExample


@dataclass
class RewardPairBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    prompt_texts: List[str]
    chosen_texts: List[str]
    rejected_texts: List[str]
    row_ids: List[str]

    def to(self, device: torch.device) -> "RewardPairBatch":
        return RewardPairBatch(
            chosen_input_ids=self.chosen_input_ids.to(device, non_blocking=True),
            chosen_attention_mask=self.chosen_attention_mask.to(device, non_blocking=True),
            rejected_input_ids=self.rejected_input_ids.to(device, non_blocking=True),
            rejected_attention_mask=self.rejected_attention_mask.to(device, non_blocking=True),
            prompt_texts=self.prompt_texts,
            chosen_texts=self.chosen_texts,
            rejected_texts=self.rejected_texts,
            row_ids=self.row_ids,
        )


@dataclass
class RewardScoringBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_texts: List[str]
    response_texts: List[str]
    row_ids: List[str]

    def to(self, device: torch.device) -> "RewardScoringBatch":
        return RewardScoringBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            prompt_texts=self.prompt_texts,
            response_texts=self.response_texts,
            row_ids=self.row_ids,
        )


class RewardPairCollator:
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

    def __call__(self, examples: Sequence[PreferenceExample]) -> RewardPairBatch:
        chosen_ids: List[torch.Tensor] = []
        rejected_ids: List[torch.Tensor] = []
        prompt_texts: List[str] = []
        chosen_texts: List[str] = []
        rejected_texts: List[str] = []
        row_ids: List[str] = []

        for ex in examples:
            chosen_ids.append(
                _tokenize_prompt_with_response(
                    tokenizer=self.tokenizer,
                    prompt_messages=ex.prompt_messages,
                    response_text=ex.chosen_text,
                    max_prompt_tokens=self.max_prompt_tokens,
                    max_response_tokens=self.max_response_tokens,
                )
            )
            rejected_ids.append(
                _tokenize_prompt_with_response(
                    tokenizer=self.tokenizer,
                    prompt_messages=ex.prompt_messages,
                    response_text=ex.rejected_text,
                    max_prompt_tokens=self.max_prompt_tokens,
                    max_response_tokens=self.max_response_tokens,
                )
            )
            prompt_texts.append(ex.prompt_text)
            chosen_texts.append(ex.chosen_text)
            rejected_texts.append(ex.rejected_text)
            row_ids.append(ex.row_id)

        max_seq_len = max(
            max(int(ids.numel()) for ids in chosen_ids),
            max(int(ids.numel()) for ids in rejected_ids),
        )
        chosen_input_ids, chosen_attention_mask = _left_pad_sequences(
            chosen_ids,
            pad_token_id=int(self.tokenizer.pad_token_id),
            max_len=max_seq_len,
        )
        rejected_input_ids, rejected_attention_mask = _left_pad_sequences(
            rejected_ids,
            pad_token_id=int(self.tokenizer.pad_token_id),
            max_len=max_seq_len,
        )
        return RewardPairBatch(
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=rejected_attention_mask,
            prompt_texts=prompt_texts,
            chosen_texts=chosen_texts,
            rejected_texts=rejected_texts,
            row_ids=row_ids,
        )


class RewardScoringCollator:
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

    def __call__(self, examples: Sequence[GenerationExample | Dict[str, object]]) -> RewardScoringBatch:
        ids_list: List[torch.Tensor] = []
        prompt_texts: List[str] = []
        response_texts: List[str] = []
        row_ids: List[str] = []
        for ex in examples:
            if isinstance(ex, GenerationExample):
                prompt_messages = ex.prompt_messages
                prompt_text = ex.prompt_text
                response_text = ex.reference_response_text or ""
                row_id = ex.row_id
            else:
                prompt_messages = list(ex["prompt_messages"])  # type: ignore[index]
                prompt_text = str(ex["prompt_text"])  # type: ignore[index]
                response_text = str(ex["response_text"])  # type: ignore[index]
                row_id = str(ex.get("row_id", len(row_ids)))  # type: ignore[union-attr]
            ids_list.append(
                _tokenize_prompt_with_response(
                    tokenizer=self.tokenizer,
                    prompt_messages=prompt_messages,
                    response_text=response_text,
                    max_prompt_tokens=self.max_prompt_tokens,
                    max_response_tokens=self.max_response_tokens,
                )
            )
            prompt_texts.append(prompt_text)
            response_texts.append(response_text)
            row_ids.append(row_id)

        input_ids, attention_mask = _left_pad_sequences(
            ids_list,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        return RewardScoringBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_texts=prompt_texts,
            response_texts=response_texts,
            row_ids=row_ids,
        )


def _tokenize_prompt_with_response(
    tokenizer: PreTrainedTokenizerBase,
    prompt_messages: Sequence[Dict[str, str]],
    response_text: str,
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> torch.Tensor:
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
        raise ValueError("Response tokenization produced zero completion tokens; cannot score reward pair.")
    if response_len > max_response_tokens:
        full_ids = full_ids[: int(prompt_ids.numel() + max_response_tokens)]
    return full_ids


def _left_pad_sequences(
    ids_list: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_len is None:
        max_len = max(int(ids.numel()) for ids in ids_list)
    batch_size = len(ids_list)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        n = int(ids.numel())
        input_ids[i, max_len - n :] = ids
        attention_mask[i, max_len - n :] = 1
    return input_ids, attention_mask
