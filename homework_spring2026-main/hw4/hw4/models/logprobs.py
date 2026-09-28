from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_per_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. input_ids/attention_mask are [B, L]; output is [B, L-1]."""
    # TODO(student): implement next-token log-probs aligned to target tokens.
    # Notation:
    # - B = batch size (number of sequences)
    # - L = tokenized sequence length including prompt, completion, and any padding
    # - V = vocabulary size
    #
    # Hugging Face model call signature to use here:
    #   out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # and then out.logits has shape [B, L, V].
    #
    # For token position t>=1, use logits at position t-1 to score target token x_t:
    #   log p(x_t | x_<t) = log_softmax(logits[:, t-1, :])[x_t].
    #
    # The naive implementation would take logits[:, :-1, :] with shape [B, L-1, V],
    # materialize ANOTHER dense [B, L-1, V] log_softmax tensor, and then gather the
    # entries for the target tokens input_ids[:, 1:].
    #
    # A more memory-efficient path is to reuse the existing logits tensor and call
    # F.cross_entropy(..., reduction='none'), because cross-entropy is exactly the
    # fused "log_softmax + gather target token + negative sign" operation.
    # Concretely:
    # - logits[:, :-1, :] has shape [B, L-1, V]
    # - targets = input_ids[:, 1:] has shape [B, L-1]
    # - flatten to [(B*(L-1)), V] and [B*(L-1)]
    # - compute per-token NLL with reduction='none'
    # - negate and reshape back to [B, L-1]
    #
    # Respect enable_grad: when enable_grad=False this function should not build an
    # autograd graph.
    raise NotImplementedError("student TODO: compute_per_token_logprobs")


def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    # TODO(student): return a float mask of shape [B, L-1] on the same device as
    # input_ids. Here input_ids and attention_mask both have shape [B, L].
    #
    # The per-token logprob tensor is indexed by t in [0, L-2], where entry t scores
    # token input_ids[:, t+1]. Therefore:
    #   mask[:, t] should be 1 iff token (t+1) belongs to the generated completion
    #   and is not padding; otherwise 0.
    # Equivalently, the FIRST completion token lives at token index prompt_input_len
    # in input_ids, which corresponds to per-token logprob index prompt_input_len - 1.
    #
    # prompt_input_len is the (padded) prompt length before completion tokens were
    # appended. You can use attention_mask to exclude padding; pad_token_id is passed
    # for convenience but a direct attention-mask-based solution is fine.
    raise NotImplementedError("student TODO: build_completion_mask")


def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions."""
    # TODO(student): implement a masked mean KL proxy. All three inputs have shape
    # [B, L-1], and mask selects only completion-token positions.
    #
    # This is an approximate / sampled KL, not an exact full-vocabulary KL at each
    # position: we only evaluate the sampled completion tokens a, then average over
    # those sampled tokens.
    #
    # Compute:
    # 1. delta = clamp(log p_ref(a) - log p_new(a), [-log_ratio_clip, log_ratio_clip])
    # 2. per_token = exp(delta) - delta - 1
    # 3. return the masked average over completion tokens
    #
    # Why this estimates KL(p_new || p_ref):
    # With delta = log(p_ref(a) / p_new(a)) and a ~ p_new,
    #   E[exp(delta)] = E[p_ref(a) / p_new(a)] = 1.
    # So
    #   E[exp(delta) - delta - 1] = -E[delta]
    #                             = E[log p_new(a) - log p_ref(a)]
    #                             = KL(p_new || p_ref).
    #
    # The clamp to [-20, 20] is for numerical stability / variance control.
    raise NotImplementedError("student TODO: approx_kl_from_logprobs")
