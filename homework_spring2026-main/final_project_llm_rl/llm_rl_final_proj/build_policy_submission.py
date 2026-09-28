from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from llm_rl_final_proj.data.ultrafeedback import GenerationExample
from llm_rl_final_proj.models.load import load_inference_model_and_tokenizer, resolve_adapter_path
from llm_rl_final_proj.offline.evaluation import generate_samples


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Generate a public-eval policy submission JSONL from a LoRA adapter.')
    ap.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--adapter_path', type=str, default='')
    ap.add_argument('--prompts_jsonl', type=str, required=True)
    ap.add_argument('--output_jsonl', type=str, required=True)
    ap.add_argument('--per_device_eval_batch_size', type=int, default=8)
    ap.add_argument('--max_prompt_tokens', type=int, default=700)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--top_p', type=float, default=1.0)
    return ap.parse_args()


def _load_generation_examples(path: Path) -> List[GenerationExample]:
    examples: List[GenerationExample] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: Dict[str, Any] = json.loads(line)
            examples.append(
                GenerationExample(
                    row_id=str(row['row_id']),
                    prompt_messages=list(row.get('prompt_messages', [])),
                    prompt_text=str(row.get('prompt_text', '')),
                    reference_response_text=row.get('reference_response_text'),
                )
            )
    return examples


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    adapter_path = resolve_adapter_path(args.adapter_path) if str(args.adapter_path).strip() else None
    examples = _load_generation_examples(Path(args.prompts_jsonl))
    if not examples:
        raise RuntimeError('No prompt rows found in prompts_jsonl.')

    loaded = load_inference_model_and_tokenizer(
        args.model_name,
        device=device,
        dtype=dtype,
        adapter_path=adapter_path,
    )
    rows = generate_samples(
        loaded.model,
        loaded.tokenizer,
        examples,
        device=device,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.per_device_eval_batch_size,
    )

    out_rows = [
        {
            'row_id': row['row_id'],
            'prompt_text': row['prompt'],
            'response_text': row['model_response'],
            'generated_num_tokens': row.get('generated_num_tokens'),
        }
        for row in rows
    ]
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')

    payload = {
        'model_name': args.model_name,
        'adapter_path': adapter_path,
        'prompt_count': len(out_rows),
        'output_jsonl': str(out_path),
        'generation_params': {
            'max_prompt_tokens': args.max_prompt_tokens,
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'per_device_eval_batch_size': args.per_device_eval_batch_size,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
