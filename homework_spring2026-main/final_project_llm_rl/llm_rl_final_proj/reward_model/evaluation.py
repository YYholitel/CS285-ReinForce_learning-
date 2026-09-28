from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_rl_final_proj.data.ultrafeedback import PreferenceExample
from llm_rl_final_proj.models.load import RewardModel
from llm_rl_final_proj.reward_model.batch import RewardPairCollator, RewardScoringCollator


def reward_model_scores(model: RewardModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    if logits.ndim == 2 and logits.shape[-1] == 1:
        return logits[:, 0]
    if logits.ndim == 1:
        return logits
    raise ValueError(f"Unexpected reward-model logits shape: {tuple(logits.shape)}")


@torch.no_grad()
def evaluate_reward_model_dataset(
    model: RewardModel,
    tokenizer,
    examples: Sequence[PreferenceExample],
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    per_device_eval_batch_size: int,
    device: torch.device,
    desc: str = "eval[reward_model]",
) -> Dict[str, float]:
    collator = RewardPairCollator(
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    loader = DataLoader(
        list(examples),
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    margin_values: List[torch.Tensor] = []
    chosen_values: List[torch.Tensor] = []
    rejected_values: List[torch.Tensor] = []
    total_examples = 0
    iterator = tqdm(loader, desc=desc, dynamic_ncols=True) if len(examples) > per_device_eval_batch_size else loader
    for batch in iterator:
        batch = batch.to(device)
        chosen_scores = reward_model_scores(
            model,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        rejected_scores = reward_model_scores(
            model,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        margin_values.append((chosen_scores - rejected_scores).detach().cpu())
        chosen_values.append(chosen_scores.detach().cpu())
        rejected_values.append(rejected_scores.detach().cpu())
        total_examples += int(chosen_scores.shape[0])
    if total_examples == 0:
        raise RuntimeError("No evaluation examples were provided.")
    margin_all = torch.cat(margin_values, dim=0)
    chosen_all = torch.cat(chosen_values, dim=0)
    rejected_all = torch.cat(rejected_values, dim=0)
    return {
        "eval/rm_pair_accuracy": float((margin_all > 0).float().mean().item()),
        "eval/rm_margin_mean": float(margin_all.mean().item()),
        "eval/rm_margin_std": float(margin_all.std(unbiased=False).item()),
        "eval/rm_chosen_score_mean": float(chosen_all.mean().item()),
        "eval/rm_rejected_score_mean": float(rejected_all.mean().item()),
        "eval/count_preference_pairs": float(total_examples),
    }


@torch.no_grad()
def score_prompt_response_pairs(
    model: RewardModel,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    per_device_batch_size: int,
    device: torch.device,
) -> List[float]:
    collator = RewardScoringCollator(
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    loader = DataLoader(
        list(rows),
        batch_size=per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    out: List[float] = []
    for batch in loader:
        batch = batch.to(device)
        scores = reward_model_scores(
            model,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        out.extend(float(x) for x in scores.detach().cpu().tolist())
    return out
