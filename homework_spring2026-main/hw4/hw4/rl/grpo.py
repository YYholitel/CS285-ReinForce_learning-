from __future__ import annotations

import math
from typing import Dict

import torch

from hw4.models.logprobs import (
    approx_kl_from_logprobs,
    compute_per_token_logprobs,
    masked_mean,
    masked_mean_per_row,
)
from hw4.rl.base import RLAlgorithm
from hw4.rollout.rollout_buffer import RolloutBatch, iter_minibatches
from hw4.utils.torch_utils import clip_grad_norm_


class GRPO(RLAlgorithm):
    """GRPO update with a PPO-style clipped surrogate over completion tokens."""

    name = "grpo"

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        cfg = self.cfg
        model.train()
        model.config.use_cache = False

        total_loss = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        total_entropy = 0.0
        n_mb = 0
        accum = 0
        skipped_empty = 0
        skipped_nonfinite = 0
        total_grad_norm = 0.0
        opt_steps = 0
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer.zero_grad(set_to_none=True)
        rng = torch.Generator(device=rollout.input_ids.device)
        rng.manual_seed(self._next_update_seed())

        for _ in range(cfg.ppo_epochs):
            for mb in iter_minibatches(
                rollout,
                cfg.minibatch_size,
                shuffle=True,
                generator=rng,
                device=next(model.parameters()).device,
            ):
                adv = mb.advantages.clamp(-cfg.adv_clip, cfg.adv_clip).detach()
                mask = mb.completion_mask
                if float(mask.sum().item()) <= 0.0:
                    skipped_empty += 1
                    continue

                # TODO(student): compute the GRPO minibatch quantities used by the
                # loss / logging code below.
                #
                # Shapes:
                # - mb.input_ids and mb.attention_mask: [B_mb, L]
                # - new_logp, mb.old_logprobs, mb.ref_logprobs, mask: [B_mb, L-1]
                # - adv: [B_mb]
                #
                # The rollout already cached:
                # - mb.old_logprobs: sampled-token log-probs under the behavior policy
                # - mb.ref_logprobs: sampled-token log-probs under the frozen reference
                # So in this update you should only recompute new_logp with the current
                # trainable policy model.
                #
                # Suggested order:
                # 1. new_logp = compute_per_token_logprobs(model, mb.input_ids, mb.attention_mask)
                # 2. log_ratio = clamp(new_logp - mb.old_logprobs, [-20, 20]) BEFORE exp
                #    for numerical stability
                # 3. ratio = exp(log_ratio)
                # 4. Broadcast advantages to per-token shape via adv.unsqueeze(1)
                # 5. Build the PPO-style unclipped / clipped token objectives:
                #      unclipped_t = ratio_t * A_i
                #      clipped_t   = clip(ratio_t, 1-clip_eps, 1+clip_eps) * A_i
                #      per_token_obj_t = min(unclipped_t, clipped_t) * mask_t
                # 6. Zero out prompt/padding positions with mask, then average over
                #    completion-token positions within each sampled completion:
                #      seq_obj_i = sum_t per_token_obj_{i,t} / (sum_t mask_{i,t} + eps)
                #    The helper masked_mean_per_row(...) imported above is useful here.
                # 7. pg_loss = - mean_i seq_obj_i
                # 8. kl = approx_kl_from_logprobs(new_logp, mb.ref_logprobs, mask)
                # 9. entropy = -masked_mean(new_logp, mask) for LOGGING ONLY
                #    (do not add an entropy term to the loss)
                # 10. clipfrac = masked fraction of completion-token positions where
                #     the PPO ratio was clipped outside [1-clip_eps, 1+clip_eps]
                raise NotImplementedError("student TODO: GRPO.update minibatch computations")

                loss = (pg_loss + cfg.kl_coef * kl) / max(1, grad_accum_steps)
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue
                loss.backward()

                accum += 1

                if (accum % max(1, grad_accum_steps)) == 0:
                    gnorm = clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    if not math.isfinite(gnorm):
                        skipped_nonfinite += 1
                        optimizer.zero_grad(set_to_none=True)
                        accum = 0
                        continue
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    total_grad_norm += float(gnorm)
                    opt_steps += 1

                total_loss += float((loss.detach() * max(1, grad_accum_steps)).item())
                total_kl += float(kl.detach().item())
                total_entropy += float(entropy.detach().item())
                total_clipfrac += float(clipfrac.detach().item())
                n_mb += 1

        if accum > 0 and (accum % max(1, grad_accum_steps)) != 0:
            gnorm = clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            if math.isfinite(gnorm):
                optimizer.step()
                total_grad_norm += float(gnorm)
                opt_steps += 1
            else:
                skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)

        denom = max(1, n_mb)
        return {
            "train/policy_loss_with_kl_penalty_mean_over_minibatches": total_loss / denom,
            "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches": total_kl / denom,
            "train/policy_token_entropy_mean_over_minibatches": total_entropy / denom,
            "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches": total_clipfrac / denom,
            "train/count_minibatches_skipped_because_completion_mask_had_no_tokens": float(skipped_empty),
            "train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients": float(skipped_nonfinite),
            "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps": total_grad_norm / max(1, opt_steps),
            "train/count_optimizer_steps_per_training_iteration": float(opt_steps),
        }
