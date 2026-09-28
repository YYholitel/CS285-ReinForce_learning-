from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
from tqdm import tqdm

from hw4.models.load import load_inference_model_and_tokenizer, resolve_adapter_path, tokenize_chat_prompts
from hw4.tasks.format_copy import FormatCopyTask


@torch.no_grad()
def make_generate_fns(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
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

        if temperature <= 0:
            do_sample = False
            temperature2 = None
        else:
            do_sample = True
            temperature2 = temperature

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature2,
            top_p=top_p,
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


def _format_seconds_compact(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.2f}h"


def main():
    ap = argparse.ArgumentParser(description="Evaluate LoRA adapter checkpoint.")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--task", type=str, default="format_copy", choices=["format_copy", "math_hard"])
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--math_hard_eval_n", type=int, default=512)
    ap.add_argument("--math_hard_eval_split", type=str, default="test_subset", choices=["test_subset", "test_full"])
    ap.add_argument("--math_hard_eval_run_full_test_set", action="store_true")
    ap.add_argument("--format_copy_eval_n", type=int, default=64)
    ap.add_argument("--format_copy_eval_seed", type=int, default=123)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    args = ap.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be >= 1, got {args.max_new_tokens}")
    if args.math_hard_eval_n < 0:
        raise ValueError(f"math_hard_eval_n must be >= 0, got {args.math_hard_eval_n}")
    if args.format_copy_eval_n < 0:
        raise ValueError(f"format_copy_eval_n must be >= 0, got {args.format_copy_eval_n}")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be >= 1, got {args.eval_batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    loaded = load_inference_model_and_tokenizer(
        args.model_name,
        device=device,
        dtype=dtype,
        adapter_path=resolve_adapter_path(args.adapter_path),
    )
    model, tokenizer = loaded.model, loaded.tokenizer

    generate_fn, generate_batch_fn = make_generate_fns(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.task == "math_hard":
        from hw4.tasks.math_hard import MathHardTask

        task = MathHardTask(seed=0, train_levels=(5,), eval_subset_size=512)
        eval_split = "test_full" if args.math_hard_eval_run_full_test_set else args.math_hard_eval_split
        pool = task._get_eval_pool(eval_split)
        eval_limit = None if eval_split == "test_full" else args.math_hard_eval_n
        planned_n = int(len(pool) if eval_limit is None else min(int(eval_limit), len(pool)))
        if planned_n <= 0:
            raise RuntimeError(f"math_hard eval planned zero examples for split={eval_split} limit={eval_limit}.")
        pbar = tqdm(total=planned_n, desc=f"eval[math_hard|{eval_split}]", dynamic_ncols=True)
        eval_start = time.perf_counter()

        def generate_with_progress(messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
            text = generate_fn(messages, max_new_tokens=max_new_tokens)
            pbar.update(1)
            return text

        def generate_batch_with_progress(
            messages_batch: List[List[Dict[str, str]]], max_new_tokens: int = 256
        ) -> List[str]:
            texts = generate_batch_fn(messages_batch, max_new_tokens=max_new_tokens)
            if len(texts) != len(messages_batch):
                raise RuntimeError(
                    "Batched eval generation must return one completion per prompt. "
                    f"got={len(texts)} expected={len(messages_batch)}"
                )
            pbar.update(len(texts))
            return texts

        print(
            f"[eval][math_hard] split={eval_split} planned_examples={planned_n} "
            f"max_new_tokens={args.max_new_tokens}"
        )
        metrics = task.evaluate(
            generate_with_progress,
            max_new_tokens=args.max_new_tokens,
            limit=eval_limit,
            split=eval_split,
            generate_batch_fn=generate_batch_with_progress,
            eval_batch_size=args.eval_batch_size,
        )
        pbar.close()
        elapsed = max(1e-6, time.perf_counter() - eval_start)
        print(
            f"[eval][math_hard] finished examples={planned_n} "
            f"elapsed={_format_seconds_compact(elapsed)} "
            f"rate={planned_n / elapsed:.2f} examples/sec"
        )
    else:
        task = FormatCopyTask(seed=0)
        metrics = task.evaluate(
            generate_fn,
            max_new_tokens=min(args.max_new_tokens, 24),
            seed=args.format_copy_eval_seed,
            n_eval=args.format_copy_eval_n,
            generate_batch_fn=generate_batch_fn,
            eval_batch_size=args.eval_batch_size,
        )

    print("=== Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
