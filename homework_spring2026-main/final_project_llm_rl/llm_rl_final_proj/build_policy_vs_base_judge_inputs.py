from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from llm_rl_final_proj.data.ultrafeedback import build_generation_examples
from llm_rl_final_proj.models.load import load_inference_model_and_tokenizer, resolve_adapter_path
from llm_rl_final_proj.offline import generate_samples, summarize_generation_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate policy and base-model responses, then package them as candidate rows for the existing judge pipeline."
    )
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--generation_split", type=str, default="test_gen")
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--generation_limit", type=int, default=128)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=700)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--summary_json", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = resolve_adapter_path(args.adapter_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    generation_examples = build_generation_examples(
        args.dataset_name,
        args.generation_split,
        limit=args.generation_limit,
    )
    if not generation_examples:
        raise RuntimeError("Generation split produced zero examples.")

    policy_loaded = load_inference_model_and_tokenizer(
        args.model_name,
        device=device,
        dtype=dtype,
        adapter_path=adapter_path,
    )
    policy_rows = generate_samples(
        policy_loaded.model,
        policy_loaded.tokenizer,
        generation_examples,
        device=device,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.per_device_eval_batch_size,
    )
    del policy_loaded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_loaded = load_inference_model_and_tokenizer(
        args.model_name,
        device=device,
        dtype=dtype,
        adapter_path=None,
    )
    base_rows = generate_samples(
        base_loaded.model,
        base_loaded.tokenizer,
        generation_examples,
        device=device,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.per_device_eval_batch_size,
    )
    del base_loaded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if len(policy_rows) != len(base_rows):
        raise RuntimeError("Policy and base generation row counts did not match.")

    packaged_rows: List[Dict[str, Any]] = []
    for policy_row, base_row in zip(policy_rows, base_rows):
        if policy_row["row_id"] != base_row["row_id"]:
            raise RuntimeError(
                f"Mismatched row_id ordering between policy and base generations: {policy_row['row_id']} vs {base_row['row_id']}"
            )
        packaged_rows.append(
            {
                "row_id": policy_row["row_id"],
                "prompt_text": policy_row["prompt"],
                "reference_response_text": policy_row.get("reference_response"),
                "analysis": {
                    "source": "policy_vs_base",
                    "policy_adapter_path": adapter_path,
                    "generation_split": args.generation_split,
                },
                "kept_candidates": [
                    {
                        "sample_index": 0,
                        "label": "policy",
                        "text": policy_row["model_response"],
                        "num_new_tokens": policy_row["generated_num_tokens"],
                    },
                    {
                        "sample_index": 1,
                        "label": "base",
                        "text": base_row["model_response"],
                        "num_new_tokens": base_row["generated_num_tokens"],
                    },
                ],
            }
        )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in packaged_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "model_name": args.model_name,
        "adapter_path": adapter_path,
        "dataset_name": args.dataset_name,
        "generation_split": args.generation_split,
        "generation_limit": len(packaged_rows),
        "generation_params": {
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
        },
        "policy_generation_metrics": summarize_generation_rows(policy_rows),
        "base_generation_metrics": summarize_generation_rows(base_rows),
        "output_jsonl": str(out_path),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
