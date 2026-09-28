from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset
import torch
from torch.utils.data import Dataset as TorchDataset


Message = Dict[str, str]


@dataclass
class PreferenceExample:
    row_id: str
    prompt_messages: List[Message]
    chosen_text: str
    rejected_text: str
    prompt_text: str
    chosen_text_full: str
    rejected_text_full: str
    score_chosen: Optional[float] = None
    score_rejected: Optional[float] = None
    avg_confidence: Optional[float] = None
    avg_preference_strength: Optional[float] = None
    avg_training_quality: Optional[float] = None


@dataclass
class GenerationExample:
    row_id: str
    prompt_messages: List[Message]
    prompt_text: str
    reference_response_text: Optional[str] = None


class UltraFeedbackPreferenceDataset(TorchDataset[PreferenceExample]):
    def __init__(self, examples: Sequence[PreferenceExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self.examples[idx]


class UltraFeedbackGenerationDataset(TorchDataset[GenerationExample]):
    def __init__(self, examples: Sequence[GenerationExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> GenerationExample:
        return self.examples[idx]


def load_ultrafeedback_split(dataset_name: str, split: str) -> Dataset:
    return load_dataset(dataset_name, split=split)


def load_ultrafeedback_dataset_dict(dataset_name: str) -> DatasetDict:
    return load_dataset(dataset_name)


def dataset_overview(dataset_name: str) -> Dict[str, Any]:
    local_root = _resolve_local_dataset_root(dataset_name)
    if local_root is not None:
        return _local_dataset_overview(local_root)
    ds = load_ultrafeedback_dataset_dict(dataset_name)
    return {
        "dataset_name": dataset_name,
        "splits": {name: int(len(split_ds)) for name, split_ds in ds.items()},
        "columns_per_split": {name: list(split_ds.column_names) for name, split_ds in ds.items()},
    }


def format_messages(messages: Sequence[Message]) -> str:
    return "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages)


def build_preference_examples(dataset_name: str, split: str, limit: int = 0) -> List[PreferenceExample]:
    local_root = _resolve_local_dataset_root(dataset_name)
    if local_root is not None:
        return _build_local_preference_examples(local_root, split, limit=limit)
    ds = load_ultrafeedback_split(dataset_name, split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[PreferenceExample] = []
    for idx, row in enumerate(ds):
        out.append(_row_to_preference_example(row=row, idx=idx))
    return out


def build_generation_examples(dataset_name: str, split: str, limit: int = 0) -> List[GenerationExample]:
    local_root = _resolve_local_dataset_root(dataset_name)
    if local_root is not None:
        return _build_local_generation_examples(local_root, split, limit=limit)
    ds = load_ultrafeedback_split(dataset_name, split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[GenerationExample] = []
    for idx, row in enumerate(ds):
        out.append(_row_to_generation_example(row=row, idx=idx))
    return out


def _row_to_preference_example(row: Dict[str, Any], idx: int) -> PreferenceExample:
    chosen_messages = _normalize_messages_like(row.get("chosen"))
    rejected_messages = _normalize_messages_like(row.get("rejected"))
    prompt_messages = _normalize_prompt_messages(row, chosen_messages, rejected_messages)

    chosen_text = _assistant_completion_from_messages(chosen_messages, prompt_messages)
    rejected_text = _assistant_completion_from_messages(rejected_messages, prompt_messages)

    row_id = _row_identifier(row, idx)
    return PreferenceExample(
        row_id=row_id,
        prompt_messages=prompt_messages,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
        prompt_text=format_messages(prompt_messages),
        chosen_text_full=format_messages(chosen_messages),
        rejected_text_full=format_messages(rejected_messages),
        score_chosen=_maybe_float(row.get("score_chosen")),
        score_rejected=_maybe_float(row.get("score_rejected")),
    )


def _row_to_generation_example(row: Dict[str, Any], idx: int) -> GenerationExample:
    prompt_messages = _normalize_generation_prompt_messages(row)
    reference_response_text = _maybe_reference_response(row)
    row_id = _row_identifier(row, idx)
    return GenerationExample(
        row_id=row_id,
        prompt_messages=prompt_messages,
        prompt_text=format_messages(prompt_messages),
        reference_response_text=reference_response_text,
    )


def _normalize_messages_like(obj: Any) -> List[Message]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [{"role": "user", "content": obj}]
    if isinstance(obj, dict):
        if "messages" in obj:
            return _normalize_messages_like(obj["messages"])
        role = str(obj.get("role", "user"))
        content = _extract_content(obj)
        return [{"role": role, "content": content}]
    if isinstance(obj, Iterable) and not isinstance(obj, (bytes, bytearray)):
        out: List[Message] = []
        for item in obj:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = _extract_content(item)
                out.append({"role": role, "content": content})
            elif isinstance(item, str):
                out.append({"role": "user", "content": item})
            else:
                out.append({"role": "user", "content": str(item)})
        return out
    return [{"role": "user", "content": str(obj)}]


def _extract_content(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                    continue
                if item.get("type") == "text" and item.get("content") is not None:
                    parts.append(str(item.get("content")))
                    continue
                parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _normalize_prompt_messages(
    row: Dict[str, Any],
    chosen_messages: Sequence[Message],
    rejected_messages: Sequence[Message],
) -> List[Message]:
    prompt_raw = row.get("prompt")
    if prompt_raw is not None:
        prompt_messages = _normalize_messages_like(prompt_raw)
        if prompt_messages:
            return prompt_messages
    common_prefix = _common_message_prefix(chosen_messages, rejected_messages)
    if common_prefix:
        return _strip_trailing_assistant(common_prefix)
    messages_raw = row.get("messages")
    if messages_raw is not None:
        return _normalize_generation_prompt_messages(row)
    raise ValueError("Could not infer prompt messages from UltraFeedback preference row.")


def _normalize_generation_prompt_messages(row: Dict[str, Any]) -> List[Message]:
    prompt_raw = row.get("prompt")
    if prompt_raw is not None:
        prompt_messages = _normalize_messages_like(prompt_raw)
        if prompt_messages:
            return prompt_messages
    messages = _normalize_messages_like(row.get("messages"))
    if not messages:
        chosen_messages = _normalize_messages_like(row.get("chosen"))
        if chosen_messages:
            return _strip_trailing_assistant(chosen_messages)
        raise ValueError("Could not infer generation prompt messages from row.")
    return _strip_trailing_assistant(messages)


def _assistant_completion_from_messages(messages: Sequence[Message], prompt_messages: Sequence[Message]) -> str:
    if not messages:
        return ""
    prefix_len = _prefix_match_length(messages, prompt_messages)
    suffix = list(messages[prefix_len:])
    if not suffix:
        suffix = [messages[-1]]
    assistant_chunks = [m.get("content", "") for m in suffix if m.get("role") == "assistant"]
    if assistant_chunks:
        return "\n\n".join(chunk for chunk in assistant_chunks if chunk)
    return messages[-1].get("content", "")


def _maybe_reference_response(row: Dict[str, Any]) -> Optional[str]:
    if row.get("chosen") is not None and row.get("rejected") is not None:
        chosen_messages = _normalize_messages_like(row.get("chosen"))
        prompt_messages = _normalize_generation_prompt_messages(row)
        return _assistant_completion_from_messages(chosen_messages, prompt_messages)
    messages = _normalize_messages_like(row.get("messages"))
    if messages and messages[-1].get("role") == "assistant":
        return messages[-1].get("content", "")
    return None


def _strip_trailing_assistant(messages: Sequence[Message]) -> List[Message]:
    out = list(messages)
    while out and out[-1].get("role") == "assistant":
        out = out[:-1]
    return out


def _common_message_prefix(a: Sequence[Message], b: Sequence[Message]) -> List[Message]:
    n = min(len(a), len(b))
    out: List[Message] = []
    for i in range(n):
        if a[i].get("role") != b[i].get("role"):
            break
        if a[i].get("content") != b[i].get("content"):
            break
        out.append(dict(a[i]))
    return out


def _prefix_match_length(full_messages: Sequence[Message], prefix_messages: Sequence[Message]) -> int:
    n = min(len(full_messages), len(prefix_messages))
    for i in range(n):
        if full_messages[i].get("role") != prefix_messages[i].get("role"):
            return i
        if full_messages[i].get("content") != prefix_messages[i].get("content"):
            return i
    return n


def _row_identifier(row: Dict[str, Any], idx: int) -> str:
    for key in ("id", "prompt_id", "source_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(idx)


def _maybe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.item())
        return None
    try:
        return float(x)
    except Exception:
        return None


def _resolve_local_dataset_root(dataset_name: str) -> Optional[Path]:
    path = Path(dataset_name).expanduser()
    if not path.exists():
        return None
    return path


def _local_dataset_overview(dataset_root: Path) -> Dict[str, Any]:
    if dataset_root.is_file():
        splits = {dataset_root.stem: dataset_root}
    else:
        splits = {p.stem: p for p in sorted(dataset_root.glob("*.jsonl"))}
    return {
        "dataset_name": str(dataset_root),
        "splits": {name: int(len(_load_local_jsonl(path))) for name, path in splits.items()},
        "columns_per_split": {
            name: sorted(list((rows[0].keys() if rows else [])))
            for name, path in splits.items()
            for rows in [_load_local_jsonl(path)]
        },
    }


def _build_local_preference_examples(dataset_root: Path, split: str, limit: int = 0) -> List[PreferenceExample]:
    rows = _load_local_rows(dataset_root, split)
    if limit > 0:
        rows = rows[: min(limit, len(rows))]
    out: List[PreferenceExample] = []
    for idx, row in enumerate(rows):
        prompt_messages = _local_prompt_messages(row)
        prompt_text = str(row.get("prompt_text", format_messages(prompt_messages)))
        chosen_text = str(row.get("chosen_text", ""))
        rejected_text = str(row.get("rejected_text", ""))
        out.append(
            PreferenceExample(
                row_id=str(row.get("row_id", _row_identifier(row, idx))),
                prompt_messages=prompt_messages,
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                prompt_text=prompt_text,
                chosen_text_full=str(row.get("chosen_text_full", chosen_text)),
                rejected_text_full=str(row.get("rejected_text_full", rejected_text)),
                score_chosen=_maybe_float(row.get("score_chosen", row.get("score"))),
                score_rejected=_maybe_float(row.get("score_rejected")),
                avg_confidence=_maybe_float(row.get("avg_confidence")),
                avg_preference_strength=_maybe_float(row.get("avg_preference_strength")),
                avg_training_quality=_maybe_float(row.get("avg_training_quality")),
            )
        )
    return out


def _build_local_generation_examples(dataset_root: Path, split: str, limit: int = 0) -> List[GenerationExample]:
    rows = _load_local_rows(dataset_root, split)
    if limit > 0:
        rows = rows[: min(limit, len(rows))]
    out: List[GenerationExample] = []
    for idx, row in enumerate(rows):
        prompt_messages = _local_prompt_messages(row)
        prompt_text = str(row.get("prompt_text", format_messages(prompt_messages)))
        reference = row.get("reference_response_text")
        if reference is None:
            reference = row.get("chosen_text")
        out.append(
            GenerationExample(
                row_id=str(row.get("row_id", _row_identifier(row, idx))),
                prompt_messages=prompt_messages,
                prompt_text=prompt_text,
                reference_response_text=None if reference is None else str(reference),
            )
        )
    return out


def _load_local_rows(dataset_root: Path, split: str) -> List[Dict[str, Any]]:
    if dataset_root.is_file():
        path = dataset_root
    else:
        path = dataset_root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Local dataset split not found: {path}")
    return _load_local_jsonl(path)


def _load_local_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _local_prompt_messages(row: Dict[str, Any]) -> List[Message]:
    if row.get("prompt_messages") is not None:
        return _normalize_messages_like(row.get("prompt_messages"))
    if row.get("prompt") is not None:
        return _normalize_messages_like(row.get("prompt"))
    prompt_text = str(row.get("prompt_text", "")).strip()
    if prompt_text:
        return [{"role": "user", "content": prompt_text}]
    raise ValueError("Could not infer prompt messages from local preference row.")
