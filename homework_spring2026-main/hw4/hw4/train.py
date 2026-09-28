from __future__ import annotations

import argparse
import json
import time
import gc
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import trange, tqdm

from hw4.config import TrainConfig
from hw4.models.load import load_lora_policy_model_and_tokenizer, tokenize_chat_prompts
from hw4.rl.base import AlgoConfig
from hw4.rl.grpo import GRPO
from hw4.rl.reinforce import Reinforce
from hw4.rollout.hf_sampler import HFSampler, SamplingConfig
from hw4.rollout.rollout_buffer import RolloutBatch
from hw4.tasks.base import TaskExample
from hw4.tasks.format_copy import FormatCopyTask
from hw4.utils.seed import set_seed
from hw4.utils.wandb_utils import WandBLogger


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Single-GPU LoRA LLM-RL training.")

    # Core
    ap.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    ap.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    ap.add_argument("--task", type=str, default=TrainConfig.task, choices=["format_copy", "math_hard"])
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)

    # Rollout
    ap.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--group_size", type=int, default=TrainConfig.group_size)
    ap.add_argument("--min_new_tokens", type=int, default=TrainConfig.min_new_tokens)
    ap.add_argument("--max_new_tokens", type=int, default=TrainConfig.max_new_tokens)
    ap.add_argument("--max_prompt_tokens", type=int, default=TrainConfig.max_prompt_tokens)
    ap.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    ap.add_argument("--top_p", type=float, default=TrainConfig.top_p)
    ap.add_argument("--top_k", type=int, default=TrainConfig.top_k)
    ap.add_argument("--repetition_penalty", type=float, default=TrainConfig.repetition_penalty)

    # RL
    ap.add_argument("--algo", type=str, default=TrainConfig.algo, choices=["reinforce", "grpo"])
    ap.add_argument("--ppo_epochs", type=int, default=TrainConfig.ppo_epochs)
    ap.add_argument("--minibatch_size", type=int, default=TrainConfig.minibatch_size)
    ap.add_argument("--clip_eps", type=float, default=TrainConfig.clip_eps)
    ap.add_argument("--kl_coef", type=float, default=TrainConfig.kl_coef)
    ap.add_argument("--max_grad_norm", type=float, default=TrainConfig.max_grad_norm)
    ap.add_argument("--adv_clip", type=float, default=TrainConfig.adv_clip)
    ap.add_argument(
        "--normalize_advantages",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.normalize_advantages,
    )

    # Optimization
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=TrainConfig.betas1)
    ap.add_argument("--betas2", type=float, default=TrainConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    ap.add_argument("--grad_accum_steps", type=int, default=TrainConfig.grad_accum_steps)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=TrainConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=TrainConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=TrainConfig.lora_bias)

    # Memory/perf
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.grad_checkpointing,
    )
    ap.add_argument(
        "--rollout_on_cpu",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.rollout_on_cpu,
    )
    ap.add_argument("--cuda_empty_cache_interval", type=int, default=TrainConfig.cuda_empty_cache_interval)

    # Logging / eval
    ap.add_argument("--wandb_project", type=str, default=TrainConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=TrainConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.wandb_enabled,
    )
    ap.add_argument("--sample_log_interval", type=int, default=TrainConfig.sample_log_interval)
    ap.add_argument(
        "--sample_markdown_log_interval",
        type=int,
        default=TrainConfig.sample_markdown_log_interval,
    )
    ap.add_argument("--sample_log_n", type=int, default=TrainConfig.sample_log_n)
    ap.add_argument("--sample_log_max_chars", type=int, default=TrainConfig.sample_log_max_chars)
    ap.add_argument("--eval_interval", type=int, default=TrainConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=TrainConfig.save_interval)
    ap.add_argument("--format_copy_eval_n", type=int, default=TrainConfig.format_copy_eval_n)
    ap.add_argument("--math_hard_eval_n", type=int, default=TrainConfig.math_hard_eval_n)
    ap.add_argument("--eval_batch_size", type=int, default=TrainConfig.eval_batch_size)

    args = ap.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        task=args.task,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        algo=args.algo,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        max_grad_norm=args.max_grad_norm,
        adv_clip=args.adv_clip,
        normalize_advantages=args.normalize_advantages,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas1=args.betas1,
        betas2=args.betas2,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
        grad_checkpointing=args.grad_checkpointing,
        rollout_on_cpu=args.rollout_on_cpu,
        cuda_empty_cache_interval=args.cuda_empty_cache_interval,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_enabled=args.wandb_enabled,
        sample_log_interval=args.sample_log_interval,
        sample_markdown_log_interval=args.sample_markdown_log_interval,
        sample_log_n=args.sample_log_n,
        sample_log_max_chars=args.sample_log_max_chars,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        format_copy_eval_n=args.format_copy_eval_n,
        math_hard_eval_n=args.math_hard_eval_n,
        eval_batch_size=args.eval_batch_size,
    )


def build_algo(cfg: TrainConfig):
    if cfg.algo == "reinforce" and cfg.ppo_epochs != 1:
        raise ValueError(
            "REINFORCE is single-pass on-policy in this codebase, so --ppo_epochs must be 1 when --algo reinforce."
        )
    acfg = AlgoConfig(
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        clip_eps=cfg.clip_eps,
        kl_coef=cfg.kl_coef,
        max_grad_norm=cfg.max_grad_norm,
        adv_clip=cfg.adv_clip,
        seed=cfg.seed,
    )
    if cfg.algo == "reinforce":
        return Reinforce(acfg)
    return GRPO(acfg)


def compute_group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    # TODO(student): implement group-relative advantage normalization.
    # rewards is a flat vector of length N = batch_size * group_size in prompt-major
    # order, so the group_size sampled completions for the same prompt are contiguous.
    #
    # IMPORTANT SHAPE CONVENTION:
    # reshape to [num_groups, group_size] (NOT [group_size, num_groups]) before
    # normalizing within each prompt's group.
    #
    # For each group g and candidate i:
    #   A_{g,i} = (r_{g,i} - mean(r_g)) / (std(r_g) + eps)
    # Use the population standard deviation within each group (PyTorch:
    # std(..., unbiased=False)), not the sample-standard-deviation correction.
    #
    # Edge cases to handle:
    # - group_size <= 1
    # - rewards.numel() not divisible by group_size
    # - near-zero within-group std: do not emit NaNs/Infs; use a stable fallback
    #   of your choice for that group
    #
    # Return a flat tensor with the same shape/order as rewards.
    raise NotImplementedError("student TODO: compute_group_advantages")


def maybe_normalize_advantages(advantages: torch.Tensor, enabled: bool, eps: float = 1e-6) -> torch.Tensor:
    # TODO(student): if enabled, z-score normalize the full advantage vector:
    #   A' = (A - mean(A)) / (std(A) + eps)
    # Again use the population standard deviation (unbiased=False).
    # Otherwise return A unchanged.
    # Keep the output shape identical to the input shape.
    raise NotImplementedError("student TODO: maybe_normalize_advantages")


def maybe_update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def count_nonfinite_params(params: List[torch.nn.Parameter]) -> int:
    bad = 0
    for p in params:
        if not torch.isfinite(p.data).all():
            bad += 1
    return bad


def _to_wandb_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return v
    if torch.is_tensor(v):
        if v.numel() == 1:
            x = float(v.detach().cpu().item())
            if math.isnan(x) or math.isinf(x):
                return str(x)
            return x
        return str(v.detach().cpu().tolist())
    return str(v)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def _format_prompt(messages: List[Dict[str, str]], max_chars: int) -> str:
    text = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages)
    return _truncate_text(text, max_chars=max_chars)


def _should_aggregate_info_metric(key: str, value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    # Raw numeric predictions/targets are better in per-sample tables than batch means.
    key_l = key.lower()
    if "predicted_number" in key_l:
        return False
    if "ground_truth_number" in key_l:
        return False
    if "target_integer_ground_truth" in key_l:
        return False
    return True


def build_rollout_example_rows(
    *,
    step: int,
    cfg: TrainConfig,
    rollout_out,
    rewards: List[float],
    advantages: torch.Tensor,
    completion_tokens: torch.Tensor,
    infos: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    n_total = len(rewards)
    n_log = min(max(0, cfg.sample_log_n), n_total)
    rows: List[Dict[str, Any]] = []
    for i in range(n_log):
        meta = rollout_out.task_metas[i]
        info = infos[i]
        gt = meta.get("gt", meta.get("target", None))
        row: Dict[str, Any] = {
            "training_step_index_zero_based": int(step),
            "task_name_for_this_sample": str(rollout_out.task_names[i]),
            "sample_index_within_logged_rows_for_this_step": int(i),
            "prompt_group_index_after_prompt_replication": int(i // max(1, cfg.group_size)),
            "candidate_index_within_group_for_same_prompt": int(i % max(1, cfg.group_size)),
            "prompt_messages_with_roles_joined_into_single_text": _format_prompt(
                rollout_out.prompt_messages[i], max_chars=cfg.sample_log_max_chars
            ),
            "question_text_from_task_metadata_if_available": _truncate_text(
                str(meta.get("question", "")), max_chars=cfg.sample_log_max_chars
            ),
            "ground_truth_numeric_answer_from_task_metadata_if_available": _to_wandb_cell(gt),
            "model_completion_text": _truncate_text(
                str(rollout_out.completion_texts[i]), max_chars=cfg.sample_log_max_chars
            ),
            "generated_completion_token_count_for_this_sample": int(completion_tokens[i].item()),
            "total_reward_used_for_policy_update_for_this_sample": float(rewards[i]),
            "advantage_used_for_policy_update_for_this_sample": float(advantages[i].item()),
        }
        for k, v in info.items():
            row[k] = _to_wandb_cell(v)
        rows.append(row)
    return rows


def build_rollout_examples_markdown(
    *,
    step: int,
    rows: List[Dict[str, Any]],
    max_chars_per_json_block: int,
) -> str:
    lines = [
        f"# Rollout Prompt/Completion/Reward Breakdown",
        f"- step_zero_based: {step}",
        f"- logged_examples: {len(rows)}",
    ]
    for i, row in enumerate(rows):
        lines.append(f"\n## example_{i}")
        blob = json.dumps(row, ensure_ascii=True, indent=2, default=str)
        if max_chars_per_json_block > 0 and len(blob) > max_chars_per_json_block:
            blob = blob[:max_chars_per_json_block] + "\n... [truncated]"
        lines.append("```json")
        lines.append(blob)
        lines.append("```")
    return "\n".join(lines)


def save_checkpoint(
    out_dir: Path,
    step: int,
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> None:
    ckpt_dir = out_dir / "checkpoints" / f"step_{step:06d}"
    adapter_dir = ckpt_dir / "adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    adapter_files = []
    adapter_total_bytes = 0
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        size_bytes = int(path.stat().st_size)
        adapter_total_bytes += size_bytes
        adapter_files.append(
            {
                "path": str(path.relative_to(ckpt_dir)),
                "size_bytes": size_bytes,
            }
        )
    (ckpt_dir / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "step": step,
                "adapter_dir_exists": adapter_dir.is_dir(),
                "optimizer_state_exists": (ckpt_dir / "optimizer.pt").is_file(),
                "adapter_file_count": len(adapter_files),
                "adapter_total_bytes": adapter_total_bytes,
                "files": adapter_files,
            },
            indent=2,
        )
    )
    (ckpt_dir / "meta.json").write_text(
        json.dumps(
            {
                "step": step,
                "algo": cfg.algo,
                "task": cfg.task,
                "model_name": cfg.model_name,
            },
            indent=2,
        )
    )


@torch.no_grad()
def make_generate_fns(model: torch.nn.Module, tokenizer, device: torch.device):
    def generate_batch(messages_batch: List[List[Dict[str, str]]], max_new_tokens: int = 256) -> List[str]:
        if not messages_batch:
            return []
        model.eval()
        input_ids, attention_mask = tokenize_chat_prompts(
            tokenizer,
            messages_batch,
            add_generation_prompt=True,
            max_prompt_tokens=None,
            device=device,
        )
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        prompt_len = int(input_ids.shape[1])
        completion_ids = out[:, prompt_len:]
        pad_id = int(tokenizer.pad_token_id)
        completions: List[str] = []
        for row in completion_ids:
            if (row == pad_id).any():
                n = int((row != pad_id).sum().item())
                row = row[:n]
            completions.append(tokenizer.decode(row, skip_special_tokens=True))
        return completions

    def generate(messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
        completions = generate_batch([messages], max_new_tokens=max_new_tokens)
        if len(completions) != 1:
            raise RuntimeError(f"Expected exactly one completion, got {len(completions)}")
        return completions[0]

    return generate, generate_batch


def build_task(cfg: TrainConfig):
    if cfg.task == "format_copy":
        return FormatCopyTask(seed=cfg.seed + 11)
    if cfg.task == "math_hard":
        from hw4.tasks.math_hard import MathHardTask

        return MathHardTask(seed=cfg.seed + 17, train_levels=(5,), eval_subset_size=512)
    raise ValueError(f"Unknown task: {cfg.task}")


def _format_seconds_compact(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.2f}h"


def main():
    cfg = parse_args()
    if cfg.steps <= 0:
        raise ValueError(f"steps must be >= 1, got {cfg.steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.group_size <= 0:
        raise ValueError(f"group_size must be >= 1, got {cfg.group_size}")
    if cfg.minibatch_size <= 0:
        raise ValueError(f"minibatch_size must be >= 1, got {cfg.minibatch_size}")
    if cfg.grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be >= 1, got {cfg.grad_accum_steps}")
    if cfg.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be >= 1, got {cfg.max_new_tokens}")
    if cfg.min_new_tokens < 0:
        raise ValueError(f"min_new_tokens must be >= 0, got {cfg.min_new_tokens}")
    if cfg.min_new_tokens > cfg.max_new_tokens:
        raise ValueError(
            f"min_new_tokens ({cfg.min_new_tokens}) must be <= max_new_tokens ({cfg.max_new_tokens})."
        )
    if cfg.max_prompt_tokens <= 0:
        raise ValueError(f"max_prompt_tokens must be >= 1, got {cfg.max_prompt_tokens}")
    if cfg.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {cfg.warmup_steps}")
    if cfg.format_copy_eval_n < 0:
        raise ValueError(f"format_copy_eval_n must be >= 0, got {cfg.format_copy_eval_n}")
    if cfg.math_hard_eval_n < 0:
        raise ValueError(f"math_hard_eval_n must be >= 0, got {cfg.math_hard_eval_n}")
    if cfg.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be >= 1, got {cfg.eval_batch_size}")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    loaded = load_lora_policy_model_and_tokenizer(
        cfg.model_name,
        device=device,
        dtype=dtype,
        grad_checkpointing=cfg.grad_checkpointing,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules.split(","),
        lora_bias=cfg.lora_bias,
    )
    model, tokenizer = loaded.model, loaded.tokenizer

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. LoRA setup failed.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )

    task = build_task(cfg)
    sampler = HFSampler(tokenizer=tokenizer, device=device)
    sampling_cfg = SamplingConfig(
        min_new_tokens=cfg.min_new_tokens,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=(cfg.temperature > 0.0),
    )
    algo = build_algo(cfg)

    logger = WandBLogger(
        project=cfg.wandb_project,
        run_name=cfg.wandb_name,
        config={
            **cfg.__dict__,
            "model_count_trainable_parameters_lora_only": loaded.trainable_params,
            "model_count_total_parameters_including_frozen_base_model": loaded.total_params,
            "lora_target_module_suffixes_that_matched_model_linear_layers": loaded.lora_target_modules,
        },
        enabled=cfg.wandb_enabled,
        local_dir=out_dir,
    )

    eval_gen_fn, eval_gen_batch_fn = make_generate_fns(model, tokenizer, device)

    def run_eval_for_task(*, eval_step: int, phase: str) -> Dict[str, float]:
        if cfg.task == "math_hard":
            eval_split = "test_subset"
            pool = task._get_eval_pool(eval_split)
            planned_n = min(cfg.math_hard_eval_n, len(pool))
            eval_kwargs = {
                "max_new_tokens": cfg.max_new_tokens,
                "limit": cfg.math_hard_eval_n,
                "split": eval_split,
                "eval_batch_size": cfg.eval_batch_size,
            }
        else:
            planned_n = int(cfg.format_copy_eval_n)
            eval_kwargs = {
                "max_new_tokens": min(cfg.max_new_tokens, 24),
                "seed": cfg.seed + 123,
                "n_eval": cfg.format_copy_eval_n,
                "eval_batch_size": cfg.eval_batch_size,
            }

        planned_n = max(0, int(planned_n))
        progress_every = max(1, min(25, planned_n // 10 if planned_n >= 10 else 1))

        tqdm.write(
            f"[eval][{cfg.task}] phase={phase} step_zero_based={eval_step} "
            f"starting evaluation over ~{planned_n} examples (progress updates every {progress_every} examples)."
        )
        eval_start = time.perf_counter()
        progress = {"done": 0, "last_log": eval_start}

        def maybe_log_progress(prev_done: int) -> None:
            done = int(progress["done"])
            now = time.perf_counter()
            crossed_interval = (done // progress_every) > (prev_done // progress_every)
            should_log = (
                prev_done == 0
                or done == planned_n
                or crossed_interval
                or (now - float(progress["last_log"]) >= 15.0 and done < planned_n)
            )
            if not should_log:
                return
            elapsed = max(1e-6, now - eval_start)
            rate = done / elapsed
            remaining = max(0, planned_n - done)
            eta = remaining / max(1e-6, rate)
            pct = 100.0 * (done / max(1, planned_n))
            tqdm.write(
                f"[eval][{cfg.task}] phase={phase} step_zero_based={eval_step} "
                f"progress={done}/{planned_n} ({pct:.1f}%) elapsed={_format_seconds_compact(elapsed)} "
                f"eta~{_format_seconds_compact(eta)}"
            )
            progress["last_log"] = now

        def generate_with_progress(messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
            text = eval_gen_fn(messages, max_new_tokens=max_new_tokens)
            prev_done = int(progress["done"])
            progress["done"] += 1
            maybe_log_progress(prev_done)
            return text

        def generate_batch_with_progress(
            messages_batch: List[List[Dict[str, str]]], max_new_tokens: int = 256
        ) -> List[str]:
            texts = eval_gen_batch_fn(messages_batch, max_new_tokens=max_new_tokens)
            if len(texts) != len(messages_batch):
                raise RuntimeError(
                    "Batched eval generation must return one completion per prompt. "
                    f"got={len(texts)} expected={len(messages_batch)}"
                )
            prev_done = int(progress["done"])
            progress["done"] += len(texts)
            maybe_log_progress(prev_done)
            return texts

        metrics = task.evaluate(
            generate_with_progress,
            generate_batch_fn=generate_batch_with_progress,
            **eval_kwargs,
        )
        eval_elapsed = max(1e-6, time.perf_counter() - eval_start)
        eval_examples_done = int(progress["done"])
        tqdm.write(
            f"[eval][{cfg.task}] phase={phase} step_zero_based={eval_step} "
            f"finished {eval_examples_done} examples in {_format_seconds_compact(eval_elapsed)} "
            f"({eval_examples_done / eval_elapsed:.2f} examples/sec)."
        )
        metrics["eval/runtime_seconds_for_last_evaluation_call"] = float(eval_elapsed)
        metrics["eval/number_of_examples_processed_in_last_evaluation_call"] = float(eval_examples_done)
        metrics["eval/examples_per_second_in_last_evaluation_call"] = float(eval_examples_done / eval_elapsed)
        return metrics

    baseline_eval_metrics = run_eval_for_task(eval_step=0, phase="baseline_before_first_rl_update")
    logger.log(baseline_eval_metrics, step=0)
    if hasattr(task, "dataset_stats"):
        logger.log(getattr(task, "dataset_stats"), step=0)

    pbar = trange(cfg.steps, desc=f"train[{cfg.algo}|{cfg.task}]")
    for step in pbar:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        step_start = time.perf_counter()
        maybe_update_warmup_lr(optimizer, cfg.lr, step, cfg.warmup_steps)

        examples = task.sample_train_batch(cfg.batch_size)
        prompt_messages = [ex.messages for ex in examples]
        task_names = [ex.task_name for ex in examples]
        task_metas = [ex.meta for ex in examples]

        rollout_out = sampler.rollout(
            policy_model=model,
            prompt_messages=prompt_messages,
            task_names=task_names,
            task_metas=task_metas,
            group_size=cfg.group_size,
            sampling=sampling_cfg,
            max_prompt_tokens=cfg.max_prompt_tokens,
            output_to_cpu=cfg.rollout_on_cpu,
        )

        rewards: List[float] = []
        reward_infos: List[Dict[str, Any]] = []
        info_accum: Dict[str, float] = {}
        for i, text in enumerate(rollout_out.completion_texts):
            ex = TaskExample(meta=rollout_out.task_metas[i], messages=[], task_name=rollout_out.task_names[i])
            r, info = task.reward(ex, text)
            rewards.append(r)
            reward_infos.append(info)
            for k, v in info.items():
                if not _should_aggregate_info_metric(k, v):
                    continue
                info_accum[k] = info_accum.get(k, 0.0) + float(v)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=rollout_out.input_ids.device)
        adv_t = compute_group_advantages(rewards_t, cfg.group_size)
        adv_t = maybe_normalize_advantages(adv_t, cfg.normalize_advantages)

        rollout_batch = RolloutBatch(
            input_ids=rollout_out.input_ids,
            attention_mask=rollout_out.attention_mask,
            completion_mask=rollout_out.completion_mask,
            old_logprobs=rollout_out.old_logprobs,
            ref_logprobs=rollout_out.ref_logprobs,
            rewards=rewards_t,
            advantages=adv_t,
            task_names=rollout_out.task_names,
            completion_texts=rollout_out.completion_texts,
        )
        if cfg.rollout_on_cpu:
            # No-op when sampler already returned CPU tensors. Kept for safety.
            rollout_batch = rollout_batch.to(torch.device("cpu"))

        stats = algo.update(
            model=model,
            optimizer=optimizer,
            rollout=rollout_batch,
            grad_accum_steps=cfg.grad_accum_steps,
        )

        bad_params = count_nonfinite_params(trainable_params)
        if bad_params > 0:
            raise RuntimeError(
                f"Detected {bad_params} non-finite trainable parameter tensors after update at step={step}."
            )
        step_seconds = max(1e-6, time.perf_counter() - step_start)

        completion_tokens = rollout_out.completion_mask.sum(dim=1).float()
        prompt_tokens = rollout_out.attention_mask[:, : rollout_out.prompt_input_len].sum(dim=1).float()
        seq_tokens_total = float(rollout_out.attention_mask.sum().item())
        hit_max_new_tokens_frac = float((completion_tokens >= float(cfg.max_new_tokens)).float().mean().item())

        with torch.no_grad():
            stats["rollout/mean_total_reward_across_all_completions_in_batch_and_groups"] = float(rewards_t.mean().item())
            stats["rollout/std_total_reward_across_all_completions_in_batch_and_groups"] = float(
                rewards_t.std(unbiased=False).item()
            )
            stats["rollout/fraction_of_completions_with_nonzero_total_reward"] = float(
                (rewards_t != 0.0).float().mean().item()
            )
            stats["rollout/mean_advantage_after_group_relative_normalization"] = float(adv_t.mean().item())
            stats["rollout/std_advantage_after_group_relative_normalization"] = float(adv_t.std(unbiased=False).item())
            stats["rollout/fraction_of_completions_with_nonzero_advantage"] = float(
                (adv_t.abs() > 1e-8).float().mean().item()
            )
            stats["rollout/indicator_all_advantages_are_zero_for_this_step"] = float((adv_t.abs().max() <= 1e-8).item())
            stats["rollout/mean_generated_completion_token_count_per_completion"] = float(completion_tokens.mean().item())
            stats["rollout/std_generated_completion_token_count_per_completion"] = float(
                completion_tokens.std(unbiased=False).item()
            )
            stats["rollout/fraction_of_completions_with_zero_generated_tokens"] = float(
                (completion_tokens <= 0).float().mean().item()
            )
            stats["rollout/fraction_of_completions_that_hit_max_new_tokens_limit"] = hit_max_new_tokens_frac
            stats["rollout/mean_prompt_token_count_per_prompt_after_tokenization"] = float(prompt_tokens.mean().item())
            stats["rollout/total_nonpadding_token_count_in_rollout_batch_including_prompt_and_completion"] = seq_tokens_total
            stats["train/current_optimizer_learning_rate"] = float(optimizer.param_groups[0]["lr"])
            stats["train/wall_clock_seconds_for_this_training_iteration"] = float(step_seconds)
            stats["train/nonpadding_rollout_tokens_processed_per_second"] = float(seq_tokens_total / step_seconds)
            if torch.cuda.is_available():
                stats["train/gpu_memory_allocated_gigabytes_current"] = float(torch.cuda.memory_allocated() / (1024**3))
                stats["train/gpu_memory_reserved_gigabytes_current"] = float(torch.cuda.memory_reserved() / (1024**3))
                stats["train/gpu_peak_memory_allocated_gigabytes_since_step_start"] = float(
                    torch.cuda.max_memory_allocated() / (1024**3)
                )
            stats["model/count_trainable_parameters_lora_only"] = float(loaded.trainable_params)
            stats["model/count_total_parameters_including_frozen_base_model"] = float(loaded.total_params)
            stats["train/count_trainable_parameter_tensors_with_nonfinite_values_after_update"] = float(bad_params)
            for k, s in info_accum.items():
                stats[k] = s / max(1, len(rewards))

        logger.log(stats, step=step)
        sample_rows: List[Dict[str, Any]] = []
        if (cfg.sample_markdown_log_interval > 0 and ((step + 1) % cfg.sample_markdown_log_interval == 0)) or (
            cfg.sample_log_interval > 0 and ((step + 1) % cfg.sample_log_interval == 0)
        ):
            sample_rows = build_rollout_example_rows(
                step=step,
                cfg=cfg,
                rollout_out=rollout_out,
                rewards=rewards,
                advantages=adv_t,
                completion_tokens=completion_tokens,
                infos=reward_infos,
            )
        if sample_rows and cfg.sample_markdown_log_interval > 0 and ((step + 1) % cfg.sample_markdown_log_interval == 0):
            logger.log(
                {
                    "samples/latest_human_readable_prompt_completion_reward_breakdown_markdown": build_rollout_examples_markdown(
                        step=step,
                        rows=sample_rows,
                        max_chars_per_json_block=max(2000, cfg.sample_log_max_chars * 8),
                    )
                },
                step=step,
            )
        if sample_rows and cfg.sample_log_interval > 0 and ((step + 1) % cfg.sample_log_interval == 0):
            if sample_rows:
                logger.log_table(f"samples/{cfg.task}_prompt_completion_reward_breakdown", sample_rows, step=step)
        pbar.set_postfix(
            {
                "reward": f"{stats['rollout/mean_total_reward_across_all_completions_in_batch_and_groups']:.3f}",
                "kl": f"{stats.get('train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches', 0.0):.3f}",
                "loss": f"{stats.get('train/policy_loss_with_kl_penalty_mean_over_minibatches', 0.0):.3f}",
            }
        )

        if cfg.eval_interval > 0 and (step + 1) % cfg.eval_interval == 0:
            eval_metrics = run_eval_for_task(eval_step=step, phase="periodic_during_training")
            logger.log(eval_metrics, step=step)

        if cfg.save_interval > 0 and (step + 1) % cfg.save_interval == 0:
            save_checkpoint(out_dir, step + 1, model, tokenizer, optimizer, cfg)

        if torch.cuda.is_available() and cfg.cuda_empty_cache_interval > 0 and (step + 1) % cfg.cuda_empty_cache_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()

    save_checkpoint(out_dir, cfg.steps, model, tokenizer, optimizer, cfg)
    final_eval_metrics = run_eval_for_task(eval_step=cfg.steps, phase="final_after_last_rl_update")
    logger.log(final_eval_metrics, step=cfg.steps)
    logger.finish()


if __name__ == "__main__":
    main()
