from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from llm_rl_final_proj.models.logprobs import compute_per_token_logprobs, masked_mean_per_row
from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.offline.batch import PreferenceBatch
from llm_rl_final_proj.utils.peft_utils import disable_adapter_if_possible


@dataclass
class SequenceScores:
    chosen_logp_sum: torch.Tensor
    rejected_logp_sum: torch.Tensor
    chosen_logp_mean: torch.Tensor
    rejected_logp_mean: torch.Tensor


@dataclass
class OfflineLossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


def compute_policy_and_reference_scores(
    model: PolicyModel,
    batch: PreferenceBatch,
    *,
    need_reference: bool,
    policy_enable_grad: bool = True,
) -> tuple[SequenceScores, SequenceScores | None]:
    policy_scores = _compute_sequence_scores(model, batch=batch, enable_grad=policy_enable_grad)
    reference_scores = None
    if need_reference:
        with torch.no_grad():
            with disable_adapter_if_possible(model):
                reference_scores = _compute_sequence_scores(model, batch=batch, enable_grad=False)
    return policy_scores, reference_scores


def compute_offline_preference_loss(
    *,
    algo: str,
    beta: float,
    policy_scores: SequenceScores,
    reference_scores: SequenceScores | None,
    example_weights: torch.Tensor | None = None,
) -> OfflineLossOutput:
    """Compute the Part 1 offline preference loss.

    The student starter only includes the required Part 1 algorithms:
      - dpo
      - ipo
      - aot

    Part 2 methods should be added by extending this function.
    """
    algo = str(algo).strip().lower()
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")

    policy_margin_sum = policy_scores.chosen_logp_sum - policy_scores.rejected_logp_sum
    policy_margin_mean = policy_scores.chosen_logp_mean - policy_scores.rejected_logp_mean

    metrics: Dict[str, float] = {
        "preference/policy_margin_sum_mean": float(policy_margin_sum.detach().mean().item()),
        "preference/policy_margin_mean_mean": float(policy_margin_mean.detach().mean().item()),
        "preference/policy_accuracy_sum": float((policy_margin_sum.detach() > 0).float().mean().item()),
        "preference/policy_accuracy_mean": float((policy_margin_mean.detach() > 0).float().mean().item()),
        "preference/policy_chosen_logp_sum_mean": float(policy_scores.chosen_logp_sum.detach().mean().item()),
        "preference/policy_rejected_logp_sum_mean": float(policy_scores.rejected_logp_sum.detach().mean().item()),
        "preference/policy_chosen_logp_mean_mean": float(policy_scores.chosen_logp_mean.detach().mean().item()),
        "preference/policy_rejected_logp_mean_mean": float(policy_scores.rejected_logp_mean.detach().mean().item()),
    }

    if algo == "dpo":
        if reference_scores is None:
            raise ValueError("DPO requires reference scores.")
        ref_margin_sum = reference_scores.chosen_logp_sum - reference_scores.rejected_logp_sum
        # TODO(student): compute the reference-corrected DPO logits.
        # Hint: compare the policy margin against the frozen reference margin.
        logits = torch.empty_like(policy_margin_sum)
        # TODO(student): replace this with the DPO logistic loss.
        losses = torch.empty_like(policy_margin_sum)
        metrics.update(
            {
                "preference/reference_margin_sum_mean": float(ref_margin_sum.detach().mean().item()),
                "preference/reference_corrected_margin_mean": float(logits.detach().mean().item()),
                "preference/reference_corrected_accuracy": float((logits.detach() > 0).float().mean().item()),
            }
        )
    elif algo == "ipo":
        if reference_scores is None:
            raise ValueError("IPO requires reference scores.")
        ref_margin_sum = reference_scores.chosen_logp_sum - reference_scores.rejected_logp_sum
        # TODO(student): compute the reference-corrected IPO logits.
        logits = torch.empty_like(policy_margin_sum)
        target_gap = 1.0 / (2.0 * beta)
        # TODO(student): implement the squared IPO target-gap objective.
        losses = torch.empty_like(policy_margin_sum)
        metrics.update(
            {
                "preference/reference_margin_sum_mean": float(ref_margin_sum.detach().mean().item()),
                "preference/reference_corrected_margin_mean": float(logits.detach().mean().item()),
                "preference/ipo_target_gap": float(target_gap),
            }
        )
    elif algo == "aot":
        if reference_scores is None:
            raise ValueError("AOT requires reference scores.")
        # TODO(student): convert policy/reference scores into chosen and rejected rewards,
        # sort both reward vectors, and apply a DPO-style logistic loss to the quantile gaps.
        chosen_rewards = torch.empty_like(policy_scores.chosen_logp_sum)
        rejected_rewards = torch.empty_like(policy_scores.rejected_logp_sum)
        quantile_gap = torch.empty_like(chosen_rewards)
        losses = torch.empty_like(chosen_rewards)
        metrics.update(
            {
                "preference/aot_chosen_reward_mean": float(chosen_rewards.detach().mean().item()),
                "preference/aot_rejected_reward_mean": float(rejected_rewards.detach().mean().item()),
                "preference/aot_quantile_gap_mean": float(quantile_gap.detach().mean().item()),
                "preference/aot_quantile_accuracy": float((quantile_gap.detach() > 0).float().mean().item()),
            }
        )
    else:
        raise ValueError(
            f"Unknown offline preference algo: {algo}. "
            "The student starter exposes only Part 1 algorithms by default. "
            "Add your Part 2 method here."
        )

    if example_weights is not None:
        weights = example_weights.to(losses.device, dtype=losses.dtype).clamp_min(1e-6)
        if losses.shape != weights.shape:
            raise ValueError(
                f"example_weights shape {tuple(weights.shape)} is incompatible with losses shape {tuple(losses.shape)}"
            )
        weighted_loss = (losses * weights).sum() / weights.sum()
        metrics["preference/example_weight_mean"] = float(weights.detach().mean().item())
        metrics["preference/example_weight_min"] = float(weights.detach().min().item())
        metrics["preference/example_weight_max"] = float(weights.detach().max().item())
    else:
        weighted_loss = losses.mean()

    metrics["preference/loss"] = float(weighted_loss.detach().item())
    return OfflineLossOutput(loss=weighted_loss, metrics=metrics)


def _compute_sequence_scores(model: PolicyModel, *, batch: PreferenceBatch, enable_grad: bool) -> SequenceScores:
    input_ids = torch.cat([batch.chosen_input_ids, batch.rejected_input_ids], dim=0)
    attention_mask = torch.cat([batch.chosen_attention_mask, batch.rejected_attention_mask], dim=0)
    response_mask = torch.cat([batch.chosen_response_mask, batch.rejected_response_mask], dim=0)

    per_token_logprobs = compute_per_token_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        enable_grad=enable_grad,
    )
    seq_logp_sum = (per_token_logprobs * response_mask).sum(dim=1)
    seq_logp_mean = masked_mean_per_row(per_token_logprobs, response_mask)

    chosen_sum, rejected_sum = seq_logp_sum.chunk(2, dim=0)
    chosen_mean, rejected_mean = seq_logp_mean.chunk(2, dim=0)
    return SequenceScores(
        chosen_logp_sum=chosen_sum,
        rejected_logp_sum=rejected_sum,
        chosen_logp_mean=chosen_mean,
        rejected_logp_mean=rejected_mean,
    )
