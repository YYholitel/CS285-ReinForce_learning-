from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from llm_rl_final_proj.data.ultrafeedback import GenerationExample, build_generation_examples, dataset_overview
from llm_rl_final_proj.models.load import (
    PolicyModel,
    RewardModel,
    load_lora_policy_model_and_tokenizer,
    load_reward_model_and_tokenizer,
)
from llm_rl_final_proj.offline.evaluation import generate_samples, summarize_generation_rows
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs
from llm_rl_final_proj.rl.base import AlgoConfig
from llm_rl_final_proj.rl.dr_grpo import DrGRPO
from llm_rl_final_proj.rl.gspo import GSPO
from llm_rl_final_proj.rl.grpo import GRPO
from llm_rl_final_proj.rl.reinforce import Reinforce
from llm_rl_final_proj.rollout.hf_sampler import HFSampler, SamplingConfig
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch
from llm_rl_final_proj.utils.hardware import (
    get_cuda_memory_metrics,
    get_hardware_metrics,
    get_model_device_metrics,
    require_cuda_if_requested,
    resolve_device_and_dtype,
)
from llm_rl_final_proj.utils.seed import set_seed
from llm_rl_final_proj.utils.wandb_utils import WandBLogger


@dataclass
class OnlineRMGRPOConfig:
    algo: str = "grpo"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_adapter_path: str = ""
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_gen"
    eval_split: str = "test_gen"
    output_dir: str = "runs/rm_grpo_default"

    seed: int = 0
    steps: int = 101
    batch_size: int = 8
    group_size: int = 4

    min_new_tokens: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    lr: float = 3e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 20
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.5

    ppo_epochs: int = 2
    minibatch_size: int = 8
    clip_eps: float = 0.1
    clip_eps_high: float = 0.0
    kl_coef: float = 0.02
    adv_clip: float = 5.0

    max_prompt_tokens: int = 700
    max_response_tokens: int = 512
    train_limit: int = 0
    eval_limit: int = 64
    reward_batch_size: int = 16

    eval_interval: int = 25
    save_interval: int = 50
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_batch_size: int = 8

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    grad_checkpointing: bool = True

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "rm_grpo"
    wandb_enabled: bool = True
    sample_log_n: int = 8
    sample_log_max_chars: int = 2500


def parse_args() -> OnlineRMGRPOConfig:
    ap = argparse.ArgumentParser(description="Train a policy online with a GRPO-family algorithm using a learned reward model.")
    ap.add_argument(
        "--algo",
        type=str,
        default=OnlineRMGRPOConfig.algo,
        choices=["grpo", "dr_grpo", "gspo"],
    )
    ap.add_argument("--model_name", type=str, default=OnlineRMGRPOConfig.model_name)
    ap.add_argument("--reward_model_name", type=str, default=OnlineRMGRPOConfig.reward_model_name)
    ap.add_argument("--reward_adapter_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default=OnlineRMGRPOConfig.dataset_name)
    ap.add_argument("--train_split", type=str, default=OnlineRMGRPOConfig.train_split)
    ap.add_argument("--eval_split", type=str, default=OnlineRMGRPOConfig.eval_split)
    ap.add_argument("--output_dir", type=str, default=OnlineRMGRPOConfig.output_dir)

    ap.add_argument("--seed", type=int, default=OnlineRMGRPOConfig.seed)
    ap.add_argument("--steps", type=int, default=OnlineRMGRPOConfig.steps)
    ap.add_argument("--batch_size", type=int, default=OnlineRMGRPOConfig.batch_size)
    ap.add_argument("--group_size", type=int, default=OnlineRMGRPOConfig.group_size)

    ap.add_argument("--min_new_tokens", type=int, default=OnlineRMGRPOConfig.min_new_tokens)
    ap.add_argument("--max_new_tokens", type=int, default=OnlineRMGRPOConfig.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=OnlineRMGRPOConfig.temperature)
    ap.add_argument("--top_p", type=float, default=OnlineRMGRPOConfig.top_p)
    ap.add_argument("--top_k", type=int, default=OnlineRMGRPOConfig.top_k)
    ap.add_argument("--repetition_penalty", type=float, default=OnlineRMGRPOConfig.repetition_penalty)

    ap.add_argument("--lr", type=float, default=OnlineRMGRPOConfig.lr)
    ap.add_argument("--weight_decay", type=float, default=OnlineRMGRPOConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=OnlineRMGRPOConfig.betas1)
    ap.add_argument("--betas2", type=float, default=OnlineRMGRPOConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=OnlineRMGRPOConfig.warmup_steps)
    ap.add_argument("--grad_accum_steps", type=int, default=OnlineRMGRPOConfig.grad_accum_steps)
    ap.add_argument("--max_grad_norm", type=float, default=OnlineRMGRPOConfig.max_grad_norm)

    ap.add_argument("--ppo_epochs", type=int, default=OnlineRMGRPOConfig.ppo_epochs)
    ap.add_argument("--minibatch_size", type=int, default=OnlineRMGRPOConfig.minibatch_size)
    ap.add_argument("--clip_eps", type=float, default=OnlineRMGRPOConfig.clip_eps)
    ap.add_argument("--clip_eps_high", type=float, default=OnlineRMGRPOConfig.clip_eps_high)
    ap.add_argument("--kl_coef", type=float, default=OnlineRMGRPOConfig.kl_coef)
    ap.add_argument("--adv_clip", type=float, default=OnlineRMGRPOConfig.adv_clip)

    ap.add_argument("--max_prompt_tokens", type=int, default=OnlineRMGRPOConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=OnlineRMGRPOConfig.max_response_tokens)
    ap.add_argument("--train_limit", type=int, default=OnlineRMGRPOConfig.train_limit)
    ap.add_argument("--eval_limit", type=int, default=OnlineRMGRPOConfig.eval_limit)
    ap.add_argument("--reward_batch_size", type=int, default=OnlineRMGRPOConfig.reward_batch_size)

    ap.add_argument("--eval_interval", type=int, default=OnlineRMGRPOConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=OnlineRMGRPOConfig.save_interval)
    ap.add_argument("--eval_max_new_tokens", type=int, default=OnlineRMGRPOConfig.eval_max_new_tokens)
    ap.add_argument("--eval_temperature", type=float, default=OnlineRMGRPOConfig.eval_temperature)
    ap.add_argument("--eval_top_p", type=float, default=OnlineRMGRPOConfig.eval_top_p)
    ap.add_argument("--eval_batch_size", type=int, default=OnlineRMGRPOConfig.eval_batch_size)

    ap.add_argument("--lora_r", type=int, default=OnlineRMGRPOConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=OnlineRMGRPOConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=OnlineRMGRPOConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=OnlineRMGRPOConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=OnlineRMGRPOConfig.lora_bias)
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMGRPOConfig.grad_checkpointing,
    )

    ap.add_argument("--wandb_project", type=str, default=OnlineRMGRPOConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=OnlineRMGRPOConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMGRPOConfig.wandb_enabled,
    )
    ap.add_argument("--sample_log_n", type=int, default=OnlineRMGRPOConfig.sample_log_n)
    ap.add_argument("--sample_log_max_chars", type=int, default=OnlineRMGRPOConfig.sample_log_max_chars)
    args = ap.parse_args()
    return OnlineRMGRPOConfig(**vars(args))


def maybe_update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def _normalize_lora_target_modules(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _sample_prompt_batch(examples: Sequence[GenerationExample], batch_size: int, rng: random.Random) -> List[GenerationExample]:
    if not examples:
        raise RuntimeError("Cannot sample prompts from an empty generation split.")
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


def _compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
    *,
    divide_by_std: bool,
) -> torch.Tensor:
    del eps
    # TODO(student): compute one scalar advantage per sampled completion by grouping rewards
    # into prompt-wise batches of size `group_size`, subtracting the group mean, and optionally
    # dividing by the group standard deviation when `divide_by_std=True`.
    raise NotImplementedError("Implement _compute_group_advantages in the student starter.")


def _build_online_algo(cfg: OnlineRMGRPOConfig):
    algo_cfg = AlgoConfig(
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        clip_eps=cfg.clip_eps,
        clip_eps_high=cfg.clip_eps_high,
        kl_coef=cfg.kl_coef,
        max_grad_norm=cfg.max_grad_norm,
        adv_clip=cfg.adv_clip,
        seed=cfg.seed,
    )
    if cfg.algo == "grpo":
        return GRPO(algo_cfg)
    if cfg.algo == "dr_grpo":
        return DrGRPO(algo_cfg)
    if cfg.algo == "gspo":
        return GSPO(algo_cfg)
    raise ValueError(f"Unsupported --algo {cfg.algo}")


def _algo_divides_advantages_by_std(algo: str) -> bool:
    # TODO(student): return True for the algorithms that use group-standard-deviation
    # normalization and False for the algorithms that intentionally avoid it.
    raise NotImplementedError("Implement _algo_divides_advantages_by_std in the student starter.")


def _normalize_completion_for_reward_scoring(text: str) -> str:
    if text.strip():
        return text
    return "[no response]"


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def _sample_rows_for_logging(
    examples: Sequence[GenerationExample],
    rows: Sequence[Dict[str, Any]],
    rm_scores: Sequence[float],
    *,
    sample_log_n: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex, row, score in list(zip(examples, rows, rm_scores))[: max(0, sample_log_n)]:
        out.append(
            {
                "row_id": ex.row_id,
                "prompt": _truncate(ex.prompt_text, max_chars),
                "reference_response": _truncate(ex.reference_response_text, max_chars),
                "model_response": _truncate(str(row.get("model_response", "")), max_chars),
                "reward_model_score": float(score),
            }
        )
    return out


def save_checkpoint(model: torch.nn.Module, cfg: OnlineRMGRPOConfig, step: int) -> None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    meta = {
        "step": step,
        "model_type": "online_policy_rm_rl",
        "algo": cfg.algo,
        "model_name": cfg.model_name,
        "reward_model_name": cfg.reward_model_name,
        "reward_adapter_path": cfg.reward_adapter_path,
        "dataset_name": cfg.dataset_name,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def evaluate_policy_with_reward_model(
    *,
    policy_model: PolicyModel,
    policy_tokenizer,
    reward_model: RewardModel,
    reward_tokenizer,
    examples: Sequence[GenerationExample],
    device: torch.device,
    max_prompt_tokens: int,
    max_response_tokens: int,
    generation_max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_batch_size: int,
) -> tuple[Dict[str, float], List[Dict[str, Any]], List[float]]:
    rows = generate_samples(
        policy_model,
        policy_tokenizer,
        examples,
        device=device,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=generation_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=generation_batch_size,
    )
    metrics = summarize_generation_rows(rows)
    scoring_rows = []
    reference_rows = []
    has_reference = True
    for ex, row in zip(examples, rows):
        scoring_rows.append(
            {
                "row_id": ex.row_id,
                "prompt_messages": ex.prompt_messages,
                "prompt_text": ex.prompt_text,
                "response_text": _normalize_completion_for_reward_scoring(str(row["model_response"])),
            }
        )
        if ex.reference_response_text:
            reference_rows.append(
                {
                    "row_id": ex.row_id,
                    "prompt_messages": ex.prompt_messages,
                    "prompt_text": ex.prompt_text,
                    "response_text": ex.reference_response_text,
                }
            )
        else:
            has_reference = False
    rm_scores = score_prompt_response_pairs(
        reward_model,
        reward_tokenizer,
        scoring_rows,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
        per_device_batch_size=generation_batch_size,
        device=device,
    )
    score_tensor = torch.tensor(rm_scores, dtype=torch.float32)
    metrics["eval/rm_score_mean_on_policy_generations"] = float(score_tensor.mean().item())
    metrics["eval/rm_score_std_on_policy_generations"] = float(score_tensor.std(unbiased=False).item())
    if has_reference and reference_rows:
        ref_scores = score_prompt_response_pairs(
            reward_model,
            reward_tokenizer,
            reference_rows,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            per_device_batch_size=generation_batch_size,
            device=device,
        )
        ref_tensor = torch.tensor(ref_scores, dtype=torch.float32)
        margin = score_tensor - ref_tensor
        metrics["eval/rm_reference_score_mean_on_dataset_reference_responses"] = float(ref_tensor.mean().item())
        metrics["eval/rm_fraction_policy_scores_above_reference"] = float((margin > 0).float().mean().item())
        metrics["eval/rm_margin_policy_minus_reference_mean"] = float(margin.mean().item())
    return metrics, rows, rm_scores


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()
    if cfg.steps <= 0:
        raise ValueError(f"--steps must be >= 1, got {cfg.steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"--batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.group_size <= 0:
        raise ValueError(f"--group_size must be >= 1, got {cfg.group_size}")
    if not cfg.reward_adapter_path:
        raise ValueError("--reward_adapter_path is required")

    if cfg.wandb_name == OnlineRMGRPOConfig.wandb_name and cfg.algo != OnlineRMGRPOConfig.algo:
        cfg.wandb_name = f"rm_{cfg.algo}"
    if cfg.output_dir == OnlineRMGRPOConfig.output_dir and cfg.algo != OnlineRMGRPOConfig.algo:
        cfg.output_dir = f"runs/rm_{cfg.algo}_default"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_online_rm_grpo_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rng = random.Random(cfg.seed)
    device, dtype = resolve_device_and_dtype()
    print(
        f"[setup] device={device} dtype={dtype} algo={cfg.algo} "
        f"policy={cfg.model_name} reward_model={cfg.reward_model_name}"
    )
    print("[setup][hardware]", json.dumps(get_hardware_metrics(device), indent=2, sort_keys=True))

    dataset_info = dataset_overview(cfg.dataset_name)
    train_examples = build_generation_examples(cfg.dataset_name, cfg.train_split, limit=cfg.train_limit)
    eval_examples = build_generation_examples(cfg.dataset_name, cfg.eval_split, limit=cfg.eval_limit)
    if not train_examples:
        raise RuntimeError("Training generation split produced zero examples.")
    if not eval_examples:
        raise RuntimeError("Evaluation generation split produced zero examples.")

    loaded_policy = load_lora_policy_model_and_tokenizer(
        cfg.model_name,
        device=device,
        dtype=dtype,
        grad_checkpointing=cfg.grad_checkpointing,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=_normalize_lora_target_modules(cfg.lora_target_modules),
        lora_bias=cfg.lora_bias,
    )
    policy_model = loaded_policy.model
    policy_tokenizer = loaded_policy.tokenizer

    loaded_reward = load_reward_model_and_tokenizer(
        cfg.reward_model_name,
        device=device,
        dtype=dtype,
        adapter_path=cfg.reward_adapter_path,
    )
    reward_model = loaded_reward.model
    reward_tokenizer = loaded_reward.tokenizer
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )
    algo = _build_online_algo(cfg)
    sampler = HFSampler(policy_tokenizer, device=device)
    sampling_cfg = SamplingConfig(
        min_new_tokens=cfg.min_new_tokens,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=cfg.temperature > 0.0,
    )

    logger = WandBLogger(
        project=cfg.wandb_project,
        run_name=cfg.wandb_name,
        config=vars(cfg),
        enabled=cfg.wandb_enabled,
        local_dir=output_dir,
    )
    logger.log(
        {
            "setup/trainable_params": float(loaded_policy.trainable_params),
            "setup/total_params": float(loaded_policy.total_params),
            "setup/trainable_fraction": float(loaded_policy.trainable_params / max(1, loaded_policy.total_params)),
            "dataset/train_examples": float(len(train_examples)),
            "dataset/eval_examples": float(len(eval_examples)),
            **{f"dataset/{k}": float(v) for k, v in dataset_info["splits"].items()},
            **get_hardware_metrics(device),
            **get_model_device_metrics(policy_model),
        },
        step=0,
    )

    def run_eval(step: int, phase: str) -> Dict[str, float]:
        metrics, rows, rm_scores = evaluate_policy_with_reward_model(
            policy_model=policy_model,
            policy_tokenizer=policy_tokenizer,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            examples=eval_examples,
            device=device,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            generation_max_new_tokens=cfg.eval_max_new_tokens,
            temperature=cfg.eval_temperature,
            top_p=cfg.eval_top_p,
            generation_batch_size=cfg.eval_batch_size,
        )
        logger.log(metrics, step=step)
        logger.log_table(
            f"samples/eval_{phase}",
            _sample_rows_for_logging(
                eval_examples,
                rows,
                rm_scores,
                sample_log_n=cfg.sample_log_n,
                max_chars=cfg.sample_log_max_chars,
            ),
            step=step,
        )
        return metrics

    print("[eval] running baseline evaluation at step=0")
    run_eval(step=0, phase="baseline")

    start_time = time.time()
    for step in range(1, cfg.steps + 1):
        maybe_update_warmup_lr(optimizer, cfg.lr, step - 1, cfg.warmup_steps)
        prompt_batch = _sample_prompt_batch(train_examples, cfg.batch_size, rng)
        rollout = sampler.rollout(
            policy_model=policy_model,
            prompt_messages=[ex.prompt_messages for ex in prompt_batch],
            task_names=["synthetic_instruction_following"] * len(prompt_batch),
            task_metas=[
                {
                    "row_id": ex.row_id,
                    "prompt_text": ex.prompt_text,
                    "reference_response_text": ex.reference_response_text,
                }
                for ex in prompt_batch
            ],
            group_size=cfg.group_size,
            sampling=sampling_cfg,
            max_prompt_tokens=cfg.max_prompt_tokens,
            output_to_cpu=False,
        )

        reward_rows = []
        for i, completion_text in enumerate(rollout.completion_texts):
            meta = rollout.task_metas[i]
            reward_rows.append(
                {
                    "row_id": f"{meta.get('row_id', i)}:{i}",
                    "prompt_messages": rollout.prompt_messages[i],
                    "prompt_text": str(meta.get("prompt_text", "")),
                    "response_text": _normalize_completion_for_reward_scoring(completion_text),
                }
            )
        reward_scores = score_prompt_response_pairs(
            reward_model,
            reward_tokenizer,
            reward_rows,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            per_device_batch_size=cfg.reward_batch_size,
            device=device,
        )
        rewards = torch.tensor(reward_scores, device=device, dtype=torch.float32)
        advantages = _compute_group_advantages(
            rewards,
            cfg.group_size,
            divide_by_std=_algo_divides_advantages_by_std(cfg.algo),
        )
        batch = RolloutBatch(
            input_ids=rollout.input_ids,
            attention_mask=rollout.attention_mask,
            completion_mask=rollout.completion_mask,
            old_logprobs=rollout.old_logprobs,
            ref_logprobs=rollout.ref_logprobs,
            rewards=rewards,
            advantages=advantages,
            task_names=rollout.task_names,
            completion_texts=rollout.completion_texts,
        )
        train_metrics = algo.update(
            policy_model,
            optimizer,
            batch,
            grad_accum_steps=cfg.grad_accum_steps,
        )
        completion_lengths = batch.completion_mask.sum(dim=1).float()
        log_metrics = {
            "rollout/reward_model_score_mean": float(rewards.mean().item()),
            "rollout/reward_model_score_std": float(rewards.std(unbiased=False).item()),
            "rollout/reward_model_score_min": float(rewards.min().item()),
            "rollout/reward_model_score_max": float(rewards.max().item()),
            "rollout/advantage_mean": float(advantages.mean().item()),
            "rollout/advantage_std": float(advantages.std(unbiased=False).item()),
            "rollout/completion_mean_tokens": float(completion_lengths.mean().item()),
            "rollout/completion_max_tokens": float(completion_lengths.max().item()),
            "rollout/count_completions": float(rewards.numel()),
            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
            "time/seconds_since_start": float(time.time() - start_time),
            **train_metrics,
            **get_cuda_memory_metrics(prefix="train"),
        }
        logger.log(log_metrics, step=step)

        should_eval = (step % cfg.eval_interval == 0) or (step == cfg.steps)
        should_save = (step % cfg.save_interval == 0) or (step == cfg.steps)
        if should_eval:
            print(f"[eval] running evaluation at step={step}")
            run_eval(step=step, phase=f"step_{step}")
        if should_save:
            print(f"[checkpoint] saving step={step}")
            save_checkpoint(policy_model, cfg, step=step)

    logger.finish()


if __name__ == "__main__":
    main()
