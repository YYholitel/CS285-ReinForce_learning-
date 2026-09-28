from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from llm_rl_final_proj.data.ultrafeedback import PreferenceExample
from llm_rl_final_proj.models.load import load_reward_model_and_tokenizer, resolve_adapter_path
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Score the public reward-model eval split and emit a Gradescope submission JSONL.')
    ap.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--adapter_path', type=str, required=True)
    ap.add_argument('--prefs_jsonl', type=str, required=True)
    ap.add_argument('--output_jsonl', type=str, required=True)
    ap.add_argument('--per_device_eval_batch_size', type=int, default=8)
    ap.add_argument('--max_prompt_tokens', type=int, default=700)
    ap.add_argument('--max_response_tokens', type=int, default=512)
    return ap.parse_args()


def _load_preference_examples(path: Path) -> List[PreferenceExample]:
    rows: List[PreferenceExample] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: Dict[str, Any] = json.loads(line)
            rows.append(
                PreferenceExample(
                    row_id=str(row['row_id']),
                    prompt_messages=list(row.get('prompt_messages', [])),
                    chosen_text=str(row.get('chosen_text', '')),
                    rejected_text=str(row.get('rejected_text', '')),
                    prompt_text=str(row.get('prompt_text', '')),
                    chosen_text_full=str(row.get('chosen_text', '')),
                    rejected_text_full=str(row.get('rejected_text', '')),
                    avg_confidence=row.get('avg_confidence'),
                    avg_preference_strength=row.get('avg_preference_strength'),
                    avg_training_quality=row.get('avg_training_quality'),
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    adapter_path = resolve_adapter_path(args.adapter_path)

    examples = _load_preference_examples(Path(args.prefs_jsonl))
    if not examples:
        raise RuntimeError('No preference rows found in prefs_jsonl.')

    loaded = load_reward_model_and_tokenizer(
        args.model_name,
        device=device,
        dtype=dtype,
        adapter_path=adapter_path,
    )
    chosen_rows = [
        {
            'row_id': ex.row_id,
            'prompt_messages': ex.prompt_messages,
            'prompt_text': ex.prompt_text,
            'response_text': ex.chosen_text,
        }
        for ex in examples
    ]
    rejected_rows = [
        {
            'row_id': ex.row_id,
            'prompt_messages': ex.prompt_messages,
            'prompt_text': ex.prompt_text,
            'response_text': ex.rejected_text,
        }
        for ex in examples
    ]
    chosen_scores = score_prompt_response_pairs(
        loaded.model,
        loaded.tokenizer,
        chosen_rows,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        per_device_batch_size=args.per_device_eval_batch_size,
        device=device,
    )
    rejected_scores = score_prompt_response_pairs(
        loaded.model,
        loaded.tokenizer,
        rejected_rows,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        per_device_batch_size=args.per_device_eval_batch_size,
        device=device,
    )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for ex, cs, rs in zip(examples, chosen_scores, rejected_scores):
            f.write(json.dumps({'row_id': ex.row_id, 'chosen_score': cs, 'rejected_score': rs}, sort_keys=True) + '\n')

    payload = {
        'model_name': args.model_name,
        'adapter_path': adapter_path,
        'count_preference_rows': len(examples),
        'output_jsonl': str(out_path),
        'scoring_params': {
            'max_prompt_tokens': args.max_prompt_tokens,
            'max_response_tokens': args.max_response_tokens,
            'per_device_eval_batch_size': args.per_device_eval_batch_size,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
