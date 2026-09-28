from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from llm_rl_final_proj.data.ultrafeedback import GenerationExample, build_generation_examples, build_preference_examples
from llm_rl_final_proj.models.load import load_inference_model_and_tokenizer, resolve_adapter_path
from llm_rl_final_proj.offline import generate_samples


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sample generations from the base model or a LoRA adapter.")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--split", type=str, default="test_gen")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--n_examples", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=700)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--compare_to_base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, also generate from the frozen base model for side-by-side comparison.",
    )
    ap.add_argument("--save_json", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    examples = _load_examples(args.dataset_name, args.split, args.n_examples)
    adapter_path = resolve_adapter_path(args.adapter_path) if args.adapter_path else None

    base_rows: Optional[List[Dict[str, Any]]] = None
    if adapter_path is None or args.compare_to_base:
        base_loaded = load_inference_model_and_tokenizer(args.model_name, device=device, dtype=dtype, adapter_path=None)
        base_rows = generate_samples(
            base_loaded.model,
            base_loaded.tokenizer,
            examples,
            device=device,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )

    adapter_rows: Optional[List[Dict[str, Any]]] = None
    if adapter_path is not None:
        adapted_loaded = load_inference_model_and_tokenizer(args.model_name, device=device, dtype=dtype, adapter_path=adapter_path)
        adapter_rows = generate_samples(
            adapted_loaded.model,
            adapted_loaded.tokenizer,
            examples,
            device=device,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )

    rows = []
    for idx, ex in enumerate(examples):
        row: Dict[str, Any] = {
            "row_id": ex.row_id,
            "prompt": ex.prompt_text,
            "dataset_reference_response": ex.reference_response_text,
        }
        if base_rows is not None:
            row["base_model_response"] = base_rows[idx]["model_response"]
        if adapter_rows is not None:
            row["adapter_response"] = adapter_rows[idx]["model_response"]
        rows.append(row)

    payload = {
        "model_name": args.model_name,
        "adapter_path": adapter_path,
        "split": args.split,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_examples(dataset_name: str, split: str, n_examples: int):
    if split.endswith("prefs"):
        pref_examples = build_preference_examples(dataset_name, split, limit=n_examples)
        return [
            GenerationExample(
                row_id=ex.row_id,
                prompt_messages=ex.prompt_messages,
                prompt_text=ex.prompt_text,
                reference_response_text=ex.chosen_text,
            )
            for ex in pref_examples
        ]
    return build_generation_examples(dataset_name, split, limit=n_examples)


if __name__ == "__main__":
    main()
