from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from llm_rl_final_proj.data.ultrafeedback import build_generation_examples, build_preference_examples
from llm_rl_final_proj.models.load import load_inference_model_and_tokenizer, resolve_adapter_path
from llm_rl_final_proj.offline import evaluate_preference_dataset, generate_samples, summarize_generation_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a base model or LoRA adapter on UltraFeedback preference data.")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--eval_split", type=str, default="test_prefs")
    ap.add_argument("--generation_split", type=str, default="test_gen")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--eval_limit", type=int, default=512)
    ap.add_argument("--generation_limit", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=700)
    ap.add_argument("--max_response_tokens", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--save_json", type=str, default=None)
    ap.add_argument(
        "--compare_to_base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, compute reference-corrected metrics against the frozen base model.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    adapter_path = resolve_adapter_path(args.adapter_path) if args.adapter_path else None
    loaded = load_inference_model_and_tokenizer(
        args.model_name,
        device=device,
        dtype=dtype,
        adapter_path=adapter_path,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    eval_examples = build_preference_examples(args.dataset_name, args.eval_split, limit=args.eval_limit)
    metrics = evaluate_preference_dataset(
        model,
        tokenizer,
        eval_examples,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        need_reference=bool(args.compare_to_base),
        device=device,
        desc=f"eval[{args.eval_split}]",
    )

    rows: List[Dict[str, str]] = []
    if args.generation_limit > 0:
        generation_examples = build_generation_examples(
            args.dataset_name,
            args.generation_split,
            limit=args.generation_limit,
        )
        rows = generate_samples(
            model,
            tokenizer,
            generation_examples,
            device=device,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.per_device_eval_batch_size,
        )
        metrics.update(summarize_generation_rows(rows))

    payload = {
        "model_name": args.model_name,
        "adapter_path": adapter_path,
        "metrics": metrics,
        "samples": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
