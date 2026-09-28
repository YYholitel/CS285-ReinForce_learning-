from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm_rl_final_proj.data.ultrafeedback import build_preference_examples
from llm_rl_final_proj.models.load import load_reward_model_and_tokenizer, resolve_adapter_path
from llm_rl_final_proj.reward_model import evaluate_reward_model_dataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a reward model or reward-model adapter on preference data.")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--eval_split", type=str, default="test_prefs")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--eval_limit", type=int, default=512)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=700)
    ap.add_argument("--max_response_tokens", type=int, default=512)
    ap.add_argument("--save_json", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    adapter_path = resolve_adapter_path(args.adapter_path) if args.adapter_path else None
    loaded = load_reward_model_and_tokenizer(
        args.model_name,
        device=device,
        dtype=dtype,
        adapter_path=adapter_path,
    )
    examples = build_preference_examples(args.dataset_name, args.eval_split, limit=args.eval_limit)
    metrics = evaluate_reward_model_dataset(
        loaded.model,
        loaded.tokenizer,
        examples,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        device=device,
        desc=f"eval[reward_model|{args.eval_split}]",
    )
    payload = {
        "model_name": args.model_name,
        "adapter_path": adapter_path,
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
