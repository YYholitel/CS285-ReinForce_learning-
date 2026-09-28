from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_rl_final_proj.config import TrainConfig
from llm_rl_final_proj.data.ultrafeedback import (
    UltraFeedbackGenerationDataset,
    UltraFeedbackPreferenceDataset,
    build_generation_examples,
    build_preference_examples,
    dataset_overview,
)
from llm_rl_final_proj.models.load import load_lora_policy_model_and_tokenizer
from llm_rl_final_proj.offline import (
    PreferenceCollator,
    compute_offline_preference_loss,
    evaluate_preference_dataset,
    generate_samples,
    summarize_generation_rows,
)
from llm_rl_final_proj.offline.losses import compute_policy_and_reference_scores
from llm_rl_final_proj.utils.hardware import (
    get_cuda_memory_metrics,
    get_hardware_metrics,
    get_model_device_metrics,
    require_cuda_if_requested,
    resolve_device_and_dtype,
)
from llm_rl_final_proj.utils.seed import set_seed
from llm_rl_final_proj.utils.wandb_utils import WandBLogger


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Offline preference optimization on UltraFeedback.")
    ap.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    ap.add_argument("--dataset_name", type=str, default=TrainConfig.dataset_name)
    ap.add_argument("--train_split", type=str, default=TrainConfig.train_split)
    ap.add_argument("--eval_split", type=str, default=TrainConfig.eval_split)
    ap.add_argument("--generation_split", type=str, default=TrainConfig.generation_split)
    ap.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)

    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument(
        "--algo",
        type=str,
        default=TrainConfig.algo,
        choices=[
            "dpo",
            "ipo",
            "aot",
        ],
    )
    ap.add_argument("--num_train_epochs", type=float, default=TrainConfig.num_train_epochs)
    ap.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)

    ap.add_argument("--per_device_train_batch_size", type=int, default=TrainConfig.per_device_train_batch_size)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=TrainConfig.per_device_eval_batch_size)
    ap.add_argument("--grad_accum_steps", type=int, default=TrainConfig.grad_accum_steps)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=TrainConfig.betas1)
    ap.add_argument("--betas2", type=float, default=TrainConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    ap.add_argument("--max_grad_norm", type=float, default=TrainConfig.max_grad_norm)

    ap.add_argument("--beta", type=float, default=TrainConfig.beta)

    ap.add_argument("--max_prompt_tokens", type=int, default=TrainConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=TrainConfig.max_response_tokens)

    ap.add_argument("--train_limit", type=int, default=TrainConfig.train_limit)
    ap.add_argument("--eval_limit", type=int, default=TrainConfig.eval_limit)
    ap.add_argument("--generation_eval_limit", type=int, default=TrainConfig.generation_eval_limit)
    ap.add_argument("--generation_eval_max_new_tokens", type=int, default=TrainConfig.generation_eval_max_new_tokens)
    ap.add_argument("--generation_eval_temperature", type=float, default=TrainConfig.generation_eval_temperature)
    ap.add_argument("--generation_eval_top_p", type=float, default=TrainConfig.generation_eval_top_p)
    ap.add_argument("--generation_eval_every", type=int, default=TrainConfig.generation_eval_every)

    ap.add_argument("--eval_interval", type=int, default=TrainConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=TrainConfig.save_interval)

    ap.add_argument("--lora_r", type=int, default=TrainConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=TrainConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=TrainConfig.lora_bias)
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.grad_checkpointing,
    )

    ap.add_argument("--wandb_project", type=str, default=TrainConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=TrainConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.wandb_enabled,
    )
    ap.add_argument("--sample_log_n", type=int, default=TrainConfig.sample_log_n)
    ap.add_argument("--sample_log_max_chars", type=int, default=TrainConfig.sample_log_max_chars)

    args = ap.parse_args()
    return TrainConfig(**vars(args))


def maybe_update_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        scale = 1.0
    else:
        scale = min(1.0, float(step + 1) / float(warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def save_checkpoint(model: torch.nn.Module, cfg: TrainConfig, step: int) -> None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    meta = {
        "step": step,
        "algo": cfg.algo,
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_lora_target_modules(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def _sample_rows_for_logging(rows: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "row_id": row.get("row_id"),
                "prompt": _truncate(str(row.get("prompt", "")), max_chars),
                "reference_response": _truncate(row.get("reference_response"), max_chars),
                "model_response": _truncate(str(row.get("model_response", "")), max_chars),
                "generated_num_tokens": row.get("generated_num_tokens"),
            }
        )
    return out


def _make_generation_markdown(rows: List[Dict[str, Any]], max_chars: int) -> str:
    parts: List[str] = []
    for i, row in enumerate(rows, start=1):
        parts.append(f"## Sample {i}")
        parts.append("### Prompt")
        parts.append(_truncate(str(row.get("prompt", "")), max_chars) or "")
        ref = row.get("reference_response")
        if ref:
            parts.append("### Dataset reference response")
            parts.append(_truncate(str(ref), max_chars) or "")
        parts.append("### Model response")
        parts.append(_truncate(str(row.get("model_response", "")), max_chars) or "")
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_train_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    device, dtype = resolve_device_and_dtype()

    print(f"[setup] device={device} dtype={dtype} algo={cfg.algo} model={cfg.model_name}")
    print(f"[setup] loading dataset={cfg.dataset_name} train_split={cfg.train_split} eval_split={cfg.eval_split}")
    print("[setup][hardware]", json.dumps(get_hardware_metrics(device), indent=2, sort_keys=True))

    dataset_info = dataset_overview(cfg.dataset_name)
    train_examples = build_preference_examples(cfg.dataset_name, cfg.train_split, limit=cfg.train_limit)
    eval_examples = build_preference_examples(cfg.dataset_name, cfg.eval_split, limit=cfg.eval_limit)
    generation_examples = build_generation_examples(cfg.dataset_name, cfg.generation_split, limit=cfg.generation_eval_limit)

    if not train_examples:
        raise RuntimeError("Training split produced zero examples.")
    if not eval_examples:
        raise RuntimeError("Evaluation split produced zero examples.")

    print(
        f"[dataset] train_examples={len(train_examples)} eval_examples={len(eval_examples)} "
        f"generation_examples={len(generation_examples)}"
    )

    loaded = load_lora_policy_model_and_tokenizer(
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

    collator = PreferenceCollator(
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
            "dataset/generation_examples": float(len(generation_examples)),
            **{f"dataset/{k}": float(v) for k, v in dataset_info["splits"].items()},
            **get_hardware_metrics(device),
            **get_model_device_metrics(model),
        },
        step=0,
    )

    need_reference = cfg.algo in {"dpo", "ipo", "aot"}

    def run_eval(step: int, phase: str) -> Dict[str, float]:
        model.eval()
        eval_metrics = evaluate_preference_dataset(
            model,
            tokenizer,
            eval_examples,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            need_reference=need_reference,
            device=device,
            desc=f"eval[{cfg.algo}|{phase}]",
        )
        eval_metrics["eval/step"] = float(step)
        should_log_generations = (
            bool(generation_examples)
            and (
                step == 0
                or phase == "final"
                or cfg.generation_eval_every <= 0
                or step % cfg.generation_eval_every == 0
            )
        )
        if should_log_generations:
            all_generation_rows = generate_samples(
                model,
                tokenizer,
                generation_examples,
                device=device,
                max_prompt_tokens=cfg.max_prompt_tokens,
                max_new_tokens=cfg.generation_eval_max_new_tokens,
                temperature=cfg.generation_eval_temperature,
                top_p=cfg.generation_eval_top_p,
                batch_size=cfg.per_device_eval_batch_size,
            )
            if all_generation_rows:
                eval_metrics.update(summarize_generation_rows(all_generation_rows))
                sample_rows = all_generation_rows[: cfg.sample_log_n]
                logger.log(
                    {
                        "samples/latest_generation_markdown": _make_generation_markdown(
                            sample_rows,
                            max_chars=cfg.sample_log_max_chars,
                        )
                    },
                    step=step,
                )
                logger.log_table(
                    "samples/generation_table",
                    _sample_rows_for_logging(sample_rows, max_chars=cfg.sample_log_max_chars),
                    step=step,
                )
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

    progress = tqdm(total=estimated_total_optimizer_steps, desc=f"train[{cfg.algo}]", dynamic_ncols=True)
    while optimizer_step < estimated_total_optimizer_steps:
        for batch in train_loader:
            batch = batch.to(device)
            policy_scores, reference_scores = compute_policy_and_reference_scores(
                model,
                batch=batch,
                need_reference=need_reference,
            )
            loss_out = compute_offline_preference_loss(
                algo=cfg.algo,
                beta=cfg.beta,
                policy_scores=policy_scores,
                reference_scores=reference_scores,
                example_weights=None,
            )
            (loss_out.loss / cfg.grad_accum_steps).backward()
            microbatch_count += 1

            should_step = (microbatch_count % cfg.grad_accum_steps == 0)
            if should_step:
                maybe_update_warmup_lr(optimizer, cfg.lr, optimizer_step, cfg.warmup_steps)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm).item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                progress.update(1)

                step_metrics: Dict[str, float] = {
                    "train/optimizer_step": float(optimizer_step),
                    "train/microbatch_count": float(microbatch_count),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/gradient_global_norm_after_clipping": grad_norm,
                    **{f"train/{k}": v for k, v in loss_out.metrics.items()},
                    **get_cuda_memory_metrics(prefix="train"),
                }
                logger.log(step_metrics, step=optimizer_step)
                progress.set_postfix(
                    loss=f"{loss_out.metrics['preference/loss']:.3f}",
                    acc=f"{loss_out.metrics['preference/policy_accuracy_sum']:.3f}",
                )

                if cfg.eval_interval > 0 and optimizer_step % cfg.eval_interval == 0:
                    run_eval(step=optimizer_step, phase="periodic")
                if cfg.save_interval > 0 and optimizer_step % cfg.save_interval == 0:
                    save_checkpoint(model, cfg, optimizer_step)
                if optimizer_step >= estimated_total_optimizer_steps:
                    break
        else:
            # End of epoch. Keep going until target optimizer steps are reached.
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
