from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_rl_final_proj.data.ultrafeedback import (
    UltraFeedbackPreferenceDataset,
    build_preference_examples,
    dataset_overview,
)
from llm_rl_final_proj.models.load import load_lora_reward_model_and_tokenizer
from llm_rl_final_proj.reward_model import RewardPairCollator, evaluate_reward_model_dataset
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
class RewardModelConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    eval_split: str = "test_prefs"
    output_dir: str = "runs/reward_model_default"

    seed: int = 0
    num_train_epochs: float = 1.0
    max_steps: int = 0

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    max_prompt_tokens: int = 700
    max_response_tokens: int = 512

    train_limit: int = 0
    eval_limit: int = 512
    eval_interval: int = 200
    save_interval: int = 200

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    grad_checkpointing: bool = True

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "reward_model"
    wandb_enabled: bool = True


def parse_args() -> RewardModelConfig:
    ap = argparse.ArgumentParser(description="Train a Bradley-Terry reward model on preference pairs.")
    ap.add_argument("--model_name", type=str, default=RewardModelConfig.model_name)
    ap.add_argument("--dataset_name", type=str, default=RewardModelConfig.dataset_name)
    ap.add_argument("--train_split", type=str, default=RewardModelConfig.train_split)
    ap.add_argument("--eval_split", type=str, default=RewardModelConfig.eval_split)
    ap.add_argument("--output_dir", type=str, default=RewardModelConfig.output_dir)

    ap.add_argument("--seed", type=int, default=RewardModelConfig.seed)
    ap.add_argument("--num_train_epochs", type=float, default=RewardModelConfig.num_train_epochs)
    ap.add_argument("--max_steps", type=int, default=RewardModelConfig.max_steps)

    ap.add_argument("--per_device_train_batch_size", type=int, default=RewardModelConfig.per_device_train_batch_size)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=RewardModelConfig.per_device_eval_batch_size)
    ap.add_argument("--grad_accum_steps", type=int, default=RewardModelConfig.grad_accum_steps)
    ap.add_argument("--lr", type=float, default=RewardModelConfig.lr)
    ap.add_argument("--weight_decay", type=float, default=RewardModelConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=RewardModelConfig.betas1)
    ap.add_argument("--betas2", type=float, default=RewardModelConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=RewardModelConfig.warmup_steps)
    ap.add_argument("--max_grad_norm", type=float, default=RewardModelConfig.max_grad_norm)

    ap.add_argument("--max_prompt_tokens", type=int, default=RewardModelConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=RewardModelConfig.max_response_tokens)

    ap.add_argument("--train_limit", type=int, default=RewardModelConfig.train_limit)
    ap.add_argument("--eval_limit", type=int, default=RewardModelConfig.eval_limit)
    ap.add_argument("--eval_interval", type=int, default=RewardModelConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=RewardModelConfig.save_interval)

    ap.add_argument("--lora_r", type=int, default=RewardModelConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=RewardModelConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=RewardModelConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=RewardModelConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=RewardModelConfig.lora_bias)
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=RewardModelConfig.grad_checkpointing,
    )

    ap.add_argument("--wandb_project", type=str, default=RewardModelConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=RewardModelConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=RewardModelConfig.wandb_enabled,
    )
    args = ap.parse_args()
    return RewardModelConfig(**vars(args))


def maybe_update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def _normalize_lora_target_modules(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def save_checkpoint(model: torch.nn.Module, cfg: RewardModelConfig, step: int) -> None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    meta = {
        "step": step,
        "model_type": "reward_model",
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _compute_pair_metrics(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> Dict[str, float]:
    # TODO(student): implement the Bradley-Terry reward-model objective.
    # `chosen_scores` and `rejected_scores` are scalar rewards for the preferred and dispreferred
    # responses in the batch. Compute:
    #   1. the per-example margin,
    #   2. the mean negative log-sigmoid loss,
    #   3. summary metrics such as pair accuracy and mean margin.
    margins = torch.empty_like(chosen_scores)
    loss = torch.empty((), device=chosen_scores.device, dtype=chosen_scores.dtype)
    return {
        "loss_tensor": loss,
        "reward_model/loss": float(loss.detach().item()),
        "reward_model/pair_accuracy": float((margins.detach() > 0).float().mean().item()),
        "reward_model/margin_mean": float(margins.detach().mean().item()),
        "reward_model/chosen_score_mean": float(chosen_scores.detach().mean().item()),
        "reward_model/rejected_score_mean": float(rejected_scores.detach().mean().item()),
    }


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_reward_model_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    device, dtype = resolve_device_and_dtype()

    print(f"[setup] device={device} dtype={dtype} model={cfg.model_name}")
    print(f"[setup] loading dataset={cfg.dataset_name} train_split={cfg.train_split} eval_split={cfg.eval_split}")
    print("[setup][hardware]", json.dumps(get_hardware_metrics(device), indent=2, sort_keys=True))

    dataset_info = dataset_overview(cfg.dataset_name)
    train_examples = build_preference_examples(cfg.dataset_name, cfg.train_split, limit=cfg.train_limit)
    eval_examples = build_preference_examples(cfg.dataset_name, cfg.eval_split, limit=cfg.eval_limit)
    if not train_examples:
        raise RuntimeError("Training split produced zero examples.")
    if not eval_examples:
        raise RuntimeError("Evaluation split produced zero examples.")

    loaded = load_lora_reward_model_and_tokenizer(
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
    model = loaded.model
    tokenizer = loaded.tokenizer

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )
    collator = RewardPairCollator(
        tokenizer,
        max_prompt_tokens=cfg.max_prompt_tokens,
        max_response_tokens=cfg.max_response_tokens,
    )
    train_loader = DataLoader(
        UltraFeedbackPreferenceDataset(train_examples),
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )
    batches_per_epoch = len(train_loader)
    if batches_per_epoch <= 0:
        raise RuntimeError("Training dataloader has zero batches.")

    estimated_total_optimizer_steps = (
        cfg.max_steps
        if cfg.max_steps > 0
        else int(math.ceil((cfg.num_train_epochs * batches_per_epoch) / max(1, cfg.grad_accum_steps)))
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
            "setup/trainable_params": float(loaded.trainable_params),
            "setup/total_params": float(loaded.total_params),
            "setup/trainable_fraction": float(loaded.trainable_params / max(1, loaded.total_params)),
            "setup/estimated_total_optimizer_steps": float(estimated_total_optimizer_steps),
            "dataset/train_examples": float(len(train_examples)),
            "dataset/eval_examples": float(len(eval_examples)),
            **{f"dataset/{k}": float(v) for k, v in dataset_info["splits"].items()},
            **get_hardware_metrics(device),
            **get_model_device_metrics(model),
        },
        step=0,
    )

    def run_eval(step: int, phase: str) -> Dict[str, float]:
        model.eval()
        eval_metrics = evaluate_reward_model_dataset(
            model,
            tokenizer,
            eval_examples,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            device=device,
            desc=f"eval[reward_model|{phase}]",
        )
        eval_metrics["eval/step"] = float(step)
        logger.log(eval_metrics, step=step)
        model.train()
        return eval_metrics

    baseline_eval = run_eval(step=0, phase="baseline")
    print("[eval][baseline]", json.dumps(baseline_eval, indent=2, sort_keys=True))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    microbatch_count = 0
    train_start = time.perf_counter()

    progress = tqdm(total=estimated_total_optimizer_steps, desc="train[reward_model]", dynamic_ncols=True)
    while optimizer_step < estimated_total_optimizer_steps:
        for batch in train_loader:
            batch = batch.to(device)
            chosen_scores = model(
                input_ids=batch.chosen_input_ids,
                attention_mask=batch.chosen_attention_mask,
                use_cache=False,
            ).logits[:, 0]
            rejected_scores = model(
                input_ids=batch.rejected_input_ids,
                attention_mask=batch.rejected_attention_mask,
                use_cache=False,
            ).logits[:, 0]
            metrics = _compute_pair_metrics(chosen_scores, rejected_scores)
            loss = metrics.pop("loss_tensor")
            (loss / cfg.grad_accum_steps).backward()
            microbatch_count += 1

            if microbatch_count % cfg.grad_accum_steps != 0:
                continue

            maybe_update_warmup_lr(optimizer, cfg.lr, optimizer_step, cfg.warmup_steps)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            progress.update(1)

            logger.log(
                {
                    "train/optimizer_step": float(optimizer_step),
                    "train/microbatch_count": float(microbatch_count),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/gradient_global_norm_after_clipping": grad_norm,
                    **{f"train/{k}": v for k, v in metrics.items()},
                    **get_cuda_memory_metrics(prefix="train"),
                },
                step=optimizer_step,
            )
            progress.set_postfix(
                loss=f"{metrics['reward_model/loss']:.3f}",
                acc=f"{metrics['reward_model/pair_accuracy']:.3f}",
            )

            if cfg.eval_interval > 0 and optimizer_step % cfg.eval_interval == 0:
                run_eval(step=optimizer_step, phase="periodic")
            if cfg.save_interval > 0 and optimizer_step % cfg.save_interval == 0:
                save_checkpoint(model, cfg, optimizer_step)
            if optimizer_step >= estimated_total_optimizer_steps:
                break
        else:
            continue
        break

    progress.close()
    save_checkpoint(model, cfg, optimizer_step)
    final_eval = run_eval(step=optimizer_step, phase="final")
    elapsed = max(1e-6, time.perf_counter() - train_start)
    logger.log(
        {
            "train/elapsed_seconds": elapsed,
            "train/optimizer_steps_completed": float(optimizer_step),
            "train/optimizer_steps_per_second": float(optimizer_step / elapsed),
        },
        step=optimizer_step,
    )
    logger.finish()

    print("[eval][final]", json.dumps(final_eval, indent=2, sort_keys=True))
    print(f"[done] optimizer_steps={optimizer_step} elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
