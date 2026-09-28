from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_rl_final_proj.data.ultrafeedback import GenerationExample, PreferenceExample, build_generation_examples
from llm_rl_final_proj.models.load import PolicyModel, tokenize_chat_prompts
from llm_rl_final_proj.offline.batch import PreferenceCollator
from llm_rl_final_proj.offline.losses import compute_policy_and_reference_scores


@torch.no_grad()
def evaluate_preference_dataset(
    model: PolicyModel,
    tokenizer,
    examples: Sequence[PreferenceExample],
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    per_device_eval_batch_size: int,
    need_reference: bool,
    device: torch.device,
    desc: str = "eval[prefs]",
) -> Dict[str, float]:
    collator = PreferenceCollator(
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

    policy_margin_sum_values: List[torch.Tensor] = []
    policy_margin_mean_values: List[torch.Tensor] = []
    ref_corrected_margin_values: List[torch.Tensor] = []

    total_examples = 0
    iterator: Iterable[Any]
    if len(examples) > per_device_eval_batch_size:
        iterator = tqdm(loader, desc=desc, dynamic_ncols=True)
    else:
        iterator = loader

    for batch in iterator:
        batch = batch.to(device)
        policy_scores, reference_scores = compute_policy_and_reference_scores(
            model,
            batch=batch,
            need_reference=need_reference,
            policy_enable_grad=False,
        )
        policy_margin_sum = policy_scores.chosen_logp_sum - policy_scores.rejected_logp_sum
        policy_margin_mean = policy_scores.chosen_logp_mean - policy_scores.rejected_logp_mean
        policy_margin_sum_values.append(policy_margin_sum.detach().cpu())
        policy_margin_mean_values.append(policy_margin_mean.detach().cpu())
        if reference_scores is not None:
            ref_margin_sum = reference_scores.chosen_logp_sum - reference_scores.rejected_logp_sum
            ref_corrected_margin_values.append((policy_margin_sum - ref_margin_sum).detach().cpu())
        total_examples += int(policy_margin_sum.shape[0])

    if total_examples == 0:
        raise RuntimeError("No evaluation examples were provided.")

    policy_margin_sum_all = torch.cat(policy_margin_sum_values, dim=0)
    policy_margin_mean_all = torch.cat(policy_margin_mean_values, dim=0)
    metrics: Dict[str, float] = {
        "eval/pref_accuracy_sum_logp": float((policy_margin_sum_all > 0).float().mean().item()),
        "eval/pref_accuracy_mean_logp": float((policy_margin_mean_all > 0).float().mean().item()),
        "eval/pref_margin_sum_logp_mean": float(policy_margin_sum_all.mean().item()),
        "eval/pref_margin_mean_logp_mean": float(policy_margin_mean_all.mean().item()),
        "eval/pref_margin_sum_logp_std": float(policy_margin_sum_all.std(unbiased=False).item()),
        "eval/pref_margin_mean_logp_std": float(policy_margin_mean_all.std(unbiased=False).item()),
        "eval/count_preference_pairs": float(total_examples),
    }
    if ref_corrected_margin_values:
        ref_corrected_margin_all = torch.cat(ref_corrected_margin_values, dim=0)
        metrics["eval/reference_corrected_pref_accuracy"] = float((ref_corrected_margin_all > 0).float().mean().item())
        metrics["eval/reference_corrected_pref_margin_mean"] = float(ref_corrected_margin_all.mean().item())
    return metrics


@torch.no_grad()
def generate_samples(
    model: PolicyModel,
    tokenizer,
    examples: Sequence[GenerationExample],
    *,
    device: torch.device,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not examples:
        return rows

    model.eval()
    for start in range(0, len(examples), batch_size):
        batch_examples = list(examples[start : start + batch_size])
        messages_batch = [ex.prompt_messages for ex in batch_examples]
        input_ids, attention_mask = tokenize_chat_prompts(
            tokenizer,
            messages_batch,
            add_generation_prompt=True,
            max_prompt_tokens=max_prompt_tokens,
            device=device,
        )
        do_sample = temperature > 0.0
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        output = model.generate(**gen_kwargs)
        prompt_len = int(input_ids.shape[1])
        completion_ids = output[:, prompt_len:]
        for ex, row_ids in zip(batch_examples, completion_ids):
            rows.append(
                {
                    "row_id": ex.row_id,
                    "prompt": ex.prompt_text,
                    "reference_response": ex.reference_response_text,
                    "model_response": _decode_completion(tokenizer, row_ids),
                    "generated_num_tokens": int((row_ids != int(tokenizer.pad_token_id)).sum().item())
                    if tokenizer.pad_token_id is not None
                    else int(row_ids.numel()),
                }
            )
    return rows


def load_fixed_generation_examples(dataset_name: str, split: str, limit: int) -> List[GenerationExample]:
    return build_generation_examples(dataset_name=dataset_name, split=split, limit=limit)


def summarize_generation_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    unique_ratios: List[float] = []
    dominant_token_fracs: List[float] = []
    empty_count = 0
    token_counts: List[int] = []

    for row in rows:
        text = str(row.get("model_response", "")).strip()
        generated_num_tokens = int(row.get("generated_num_tokens", 0) or 0)
        if not text:
            token_counts.append(float(generated_num_tokens))
            empty_count += 1
            unique_ratios.append(0.0)
            dominant_token_fracs.append(1.0)
            continue

        tokens = text.split()
        if not tokens:
            token_counts.append(float(generated_num_tokens))
            empty_count += 1
            unique_ratios.append(0.0)
            dominant_token_fracs.append(1.0)
            continue

        token_counts.append(float(generated_num_tokens or len(tokens)))
        lowered = [tok.lower() for tok in tokens]
        counts = Counter(lowered)
        unique_ratios.append(len(counts) / max(1, len(lowered)))
        dominant_token_fracs.append(max(counts.values()) / max(1, len(lowered)))

    n = float(len(rows))
    mean_unique_ratio = sum(unique_ratios) / n
    mean_dominant_frac = sum(dominant_token_fracs) / n
    return {
        "eval/generation_count": n,
        "eval/generation_mean_num_tokens": float(sum(token_counts) / n),
        "eval/generation_mean_unique_token_ratio": float(mean_unique_ratio),
        "eval/generation_min_unique_token_ratio": float(min(unique_ratios)),
        "eval/generation_mean_dominant_token_fraction": float(mean_dominant_frac),
        "eval/generation_max_dominant_token_fraction": float(max(dominant_token_fracs)),
        "eval/generation_fraction_low_unique_ratio_lt_0.1": float(sum(r < 0.1 for r in unique_ratios) / n),
        "eval/generation_fraction_dominant_token_gt_0.5": float(sum(r > 0.5 for r in dominant_token_fracs) / n),
        "eval/generation_fraction_empty": float(empty_count / n),
    }


def _decode_completion(tokenizer, token_ids: torch.Tensor) -> str:
    row = token_ids
    pad_id = tokenizer.pad_token_id
    if pad_id is not None and (row == pad_id).any():
        n = int((row != pad_id).sum().item())
        row = row[:n]
    return tokenizer.decode(row, skip_special_tokens=True)
