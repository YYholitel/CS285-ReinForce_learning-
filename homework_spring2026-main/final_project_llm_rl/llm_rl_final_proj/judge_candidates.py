from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_JUDGE_MODEL = "gpt-5.4"
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXCLUSION_TAGS = [
    "explicit_sexual_content",
    "medical_mental_health_or_crisis",
    "legal_or_financial_advice",
    "copyright_or_fanfic",
    "garbled_or_underspecified",
    "high_factual_uncertainty",
    "all_candidates_weak",
]

JUDGE_INSTRUCTIONS = (
    "You are grading multiple candidate responses to the same user prompt. "
    "Select the best and worst response among the candidates. "
    "Prefer answers that are directly responsive, helpful, correct when factual, coherent, and well-matched to the prompt. "
    "For creative prompts, reward originality, style fit, and completeness. "
    "Penalize refusals, AI self-reference, meta-discussion about capabilities, unnecessary clarification requests, obvious incompleteness, and generic filler. "
    "Do not reward length by itself. If all responses are weak, still choose the relatively best and worst. "
    "Separately decide whether this prompt-plus-candidate set is usable for preference training. "
    "Be conservative. Default to usable_for_training=false unless the example is clearly good training data. "
    "Use exclusion_tags to explain why an example should be excluded. "
    "Tag explicit sexual content, medical/mental-health/crisis advice, legal or financial advice, copyright/fanfic prompts, garbled or underspecified prompts, high factual uncertainty, and cases where all candidates are too weak."
)


@dataclass(frozen=True)
class JudgeConfig:
    api_url: str
    api_key: str
    judge_model: str
    reasoning_effort: str
    verbosity: str
    max_output_tokens: int
    timeout_seconds: float
    max_retries: int


def _normalize_reasoning_effort(model: str, effort: str) -> str:
    model_l = model.lower()
    if effort == "none" and model_l.startswith("gpt-5-mini"):
        return "minimal"
    return effort


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Judge filtered candidate responses with the OpenAI Responses API.")
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--summary_json", type=str, default=None)
    ap.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    ap.add_argument("--n_rows", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reasoning_effort", type=str, default="none", choices=["none", "minimal", "low", "medium", "high"])
    ap.add_argument("--verbosity", type=str, default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max_output_tokens", type=int, default=220)
    ap.add_argument("--timeout_seconds", type=float, default=120.0)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--max_workers", type=int, default=4)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not set in the environment.")
    if args.max_workers <= 0:
        raise ValueError("--max_workers must be >= 1")

    rows = _load_jsonl(Path(args.input_jsonl))
    if not rows:
        raise RuntimeError("No rows were found in the input JSONL.")

    rng = random.Random(args.seed)
    selected = list(rows)
    if args.shuffle:
        rng.shuffle(selected)
    if args.offset > 0:
        selected = selected[args.offset :]
    if args.n_rows > 0:
        selected = selected[: args.n_rows]
    if not selected:
        raise RuntimeError("No rows remained after applying offset/n_rows.")

    reasoning_effort = _normalize_reasoning_effort(args.judge_model, args.reasoning_effort)

    cfg = JudgeConfig(
        api_url=args.api_url,
        api_key=api_key,
        judge_model=args.judge_model,
        reasoning_effort=reasoning_effort,
        verbosity=args.verbosity,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    indexed_rows = list(enumerate(selected))
    results_by_index: Dict[int, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_index = {
            executor.submit(_judge_row_safe, idx, row, cfg, args.seed): idx
            for idx, row in indexed_rows
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            results_by_index[idx] = future.result()

    ordered_results = [results_by_index[i] for i in range(len(indexed_rows))]
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in ordered_results:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = _build_summary(
        ordered_results,
        selection={
            "input_jsonl": args.input_jsonl,
            "n_rows": args.n_rows,
            "offset": args.offset,
            "shuffle": bool(args.shuffle),
            "seed": args.seed,
        },
        config={
            "judge_model": args.judge_model,
            "reasoning_effort": reasoning_effort,
            "verbosity": args.verbosity,
            "max_output_tokens": args.max_output_tokens,
            "max_workers": args.max_workers,
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
        },
    )
    text = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
    print(text)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text, encoding="utf-8")


def _judge_row_safe(index: int, row: Dict[str, Any], cfg: JudgeConfig, seed: int) -> Dict[str, Any]:
    try:
        return _judge_row(index=index, row=row, cfg=cfg, seed=seed)
    except Exception as exc:
        return {
            "status": "error",
            "row_id": row.get("row_id"),
            "score": row.get("score"),
            "prompt_text": row.get("prompt_text"),
            "judge_model": cfg.judge_model,
            "kept_candidates": row.get("kept_candidates", []),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _judge_row(*, index: int, row: Dict[str, Any], cfg: JudgeConfig, seed: int) -> Dict[str, Any]:
    kept_candidates = list(row.get("kept_candidates", []))
    if len(kept_candidates) < 2:
        raise ValueError("Need at least two kept candidates to judge a row.")
    if len(kept_candidates) > len(LABELS):
        raise ValueError(f"Too many candidates ({len(kept_candidates)}); max supported is {len(LABELS)}")

    order1 = _build_first_order(row_id=str(row.get("row_id", index)), n=len(kept_candidates), seed=seed)
    order2 = list(reversed(order1))
    if order2 == order1 and len(order1) > 1:
        order2 = order1[1:] + order1[:1]

    pass1 = _judge_once(
        prompt_text=str(row.get("prompt_text", "")),
        ordered_candidates=[kept_candidates[i] for i in order1],
        cfg=cfg,
    )
    pass2 = _judge_once(
        prompt_text=str(row.get("prompt_text", "")),
        ordered_candidates=[kept_candidates[i] for i in order2],
        cfg=cfg,
    )

    mapped1 = _map_judgment_to_sample_indices(pass1, [kept_candidates[i] for i in order1])
    mapped2 = _map_judgment_to_sample_indices(pass2, [kept_candidates[i] for i in order2])

    best_agree = mapped1["best_sample_index"] == mapped2["best_sample_index"]
    worst_agree = mapped1["worst_sample_index"] == mapped2["worst_sample_index"]
    pair_agree = best_agree and worst_agree
    avg_confidence = float(statistics.fmean([pass1["confidence"], pass2["confidence"]]))
    avg_preference_strength = float(
        statistics.fmean([pass1["preference_strength"], pass2["preference_strength"]])
    )
    avg_training_quality = float(
        statistics.fmean([pass1["training_quality"], pass2["training_quality"]])
    )
    usable_both_passes = bool(pass1["usable_for_training"] and pass2["usable_for_training"])

    return {
        "status": "ok",
        "row_id": row.get("row_id"),
        "score": row.get("score"),
        "prompt_text": row.get("prompt_text"),
        "analysis": row.get("analysis"),
        "judge_model": cfg.judge_model,
        "num_kept_candidates": len(kept_candidates),
        "kept_candidates": kept_candidates,
        "pass1": {
            "ordering_sample_indices": [int(c["sample_index"]) for c in [kept_candidates[i] for i in order1]],
            "judgment": pass1,
            **mapped1,
        },
        "pass2": {
            "ordering_sample_indices": [int(c["sample_index"]) for c in [kept_candidates[i] for i in order2]],
            "judgment": pass2,
            **mapped2,
        },
        "best_agree": best_agree,
        "worst_agree": worst_agree,
        "pair_agree": pair_agree,
        "avg_confidence": avg_confidence,
        "avg_preference_strength": avg_preference_strength,
        "avg_training_quality": avg_training_quality,
        "usable_both_passes": usable_both_passes,
    }


def _build_first_order(*, row_id: str, n: int, seed: int) -> List[int]:
    order = list(range(n))
    salt = f"{seed}:{row_id}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(salt).digest()[:8], "big")
    random.Random(derived).shuffle(order)
    return order


def _judge_once(*, prompt_text: str, ordered_candidates: Sequence[Dict[str, Any]], cfg: JudgeConfig) -> Dict[str, Any]:
    labels = list(LABELS[: len(ordered_candidates)])
    label_to_candidate = {label: candidate for label, candidate in zip(labels, ordered_candidates)}
    schema = {
        "type": "object",
        "properties": {
            "best_label": {"type": "string", "enum": labels},
            "worst_label": {"type": "string", "enum": labels},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "preference_strength": {"type": "integer", "minimum": 1, "maximum": 5},
            "usable_for_training": {"type": "boolean"},
            "training_quality": {"type": "integer", "minimum": 1, "maximum": 5},
            "exclusion_tags": {
                "type": "array",
                "items": {"type": "string", "enum": EXCLUSION_TAGS},
                "maxItems": len(EXCLUSION_TAGS),
            },
            "short_reason": {"type": "string", "minLength": 1, "maxLength": 280},
            "usable_reason": {"type": "string", "minLength": 1, "maxLength": 280},
        },
        "required": [
            "best_label",
            "worst_label",
            "confidence",
            "preference_strength",
            "usable_for_training",
            "training_quality",
            "exclusion_tags",
            "short_reason",
            "usable_reason",
        ],
        "additionalProperties": False,
    }
    user_input = _build_judge_input(prompt_text=prompt_text, labels=labels, ordered_candidates=ordered_candidates)
    payload = {
        "model": cfg.judge_model,
        "store": False,
        "instructions": JUDGE_INSTRUCTIONS,
        "input": user_input,
        "reasoning": {"effort": cfg.reasoning_effort},
        "text": {
            "verbosity": cfg.verbosity,
            "format": {
                "type": "json_schema",
                "name": "candidate_judgment",
                "strict": True,
                "schema": schema,
            },
        },
        "max_output_tokens": cfg.max_output_tokens,
    }
    response_json = _post_with_retries(cfg=cfg, payload=payload)
    raw_text = _extract_output_text(response_json)
    judgment = json.loads(raw_text)
    judgment = _normalize_judgment(judgment)
    judgment = _coerce_degenerate_judgment(judgment, label_to_candidate)
    _validate_judgment(judgment, label_to_candidate)
    judgment["response_id"] = response_json.get("id")
    judgment["usage"] = response_json.get("usage")
    return judgment


def _build_judge_input(*, prompt_text: str, labels: Sequence[str], ordered_candidates: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "User prompt:",
        prompt_text.strip(),
        "",
        "Candidate responses:",
    ]
    for label, candidate in zip(labels, ordered_candidates):
        lines.append(f"[{label}] (sample_index={int(candidate['sample_index'])})")
        lines.append(str(candidate["text"]).strip())
        lines.append("")
    lines.append("Return only the JSON object that matches the schema.")
    return "\n".join(lines)


def _map_judgment_to_sample_indices(judgment: Dict[str, Any], ordered_candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = list(LABELS[: len(ordered_candidates)])
    label_to_candidate = {label: candidate for label, candidate in zip(labels, ordered_candidates)}
    best = label_to_candidate[judgment["best_label"]]
    worst = label_to_candidate[judgment["worst_label"]]
    return {
        "best_sample_index": int(best["sample_index"]),
        "worst_sample_index": int(worst["sample_index"]),
    }


def _normalize_judgment(judgment: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(judgment)
    raw_tags = normalized.get("exclusion_tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError("Judge returned a non-list exclusion_tags field.")
    exclusion_tags: List[str] = []
    seen = set()
    for tag in raw_tags:
        if tag in seen:
            continue
        exclusion_tags.append(tag)
        seen.add(tag)
    normalized["exclusion_tags"] = exclusion_tags
    if exclusion_tags:
        normalized["usable_for_training"] = False
        normalized["training_quality"] = min(int(normalized["training_quality"]), 2)
    return normalized


def _coerce_degenerate_judgment(
    judgment: Dict[str, Any], label_to_candidate: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    normalized = dict(judgment)
    if normalized["best_label"] != normalized["worst_label"]:
        return normalized
    labels = list(label_to_candidate.keys())
    replacement_worst = next((label for label in labels if label != normalized["best_label"]), None)
    if replacement_worst is None:
        return normalized
    tags = list(normalized.get("exclusion_tags", []))
    if "all_candidates_weak" not in tags:
        tags.append("all_candidates_weak")
    normalized["worst_label"] = replacement_worst
    normalized["exclusion_tags"] = tags
    normalized["usable_for_training"] = False
    normalized["training_quality"] = min(int(normalized["training_quality"]), 1)
    short_reason = str(normalized.get("short_reason", "")).strip()
    usable_reason = str(normalized.get("usable_reason", "")).strip()
    normalized["short_reason"] = (
        short_reason + " " if short_reason else ""
    ) + "The judge collapsed best and worst onto the same candidate, so this row was coerced into an excluded weak-example comparison."
    normalized["usable_reason"] = (
        usable_reason + " " if usable_reason else ""
    ) + "This row was excluded because the judge did not identify a stable distinct best and worst candidate."
    return normalized


def _validate_judgment(judgment: Dict[str, Any], label_to_candidate: Dict[str, Dict[str, Any]]) -> None:
    if judgment["best_label"] == judgment["worst_label"]:
        raise ValueError("Judge returned the same label for best and worst.")
    if judgment["best_label"] not in label_to_candidate:
        raise ValueError(f"Unknown best label: {judgment['best_label']}")
    if judgment["worst_label"] not in label_to_candidate:
        raise ValueError(f"Unknown worst label: {judgment['worst_label']}")
    exclusion_tags = list(judgment.get("exclusion_tags", []))
    invalid_tags = [tag for tag in exclusion_tags if tag not in EXCLUSION_TAGS]
    if invalid_tags:
        raise ValueError(f"Unknown exclusion tags: {invalid_tags}")


def _post_with_retries(*, cfg: JudgeConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            return _post_json(
                url=cfg.api_url,
                api_key=cfg.api_key,
                payload=payload,
                timeout_seconds=cfg.timeout_seconds,
            )
        except Exception:
            if attempt >= cfg.max_retries:
                raise
            delay = min(30.0, 1.5 ** attempt) + random.random() * 0.25
            time.sleep(delay)


def _post_json(*, url: str, api_key: str, payload: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {exc}") from exc
    return json.loads(raw)


def _extract_output_text(response_json: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = str(content.get("text", ""))
                if text:
                    chunks.append(text)
    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError(f"No output_text content found in response: {json.dumps(response_json)[:1000]}")
    return text


def _build_summary(rows: Sequence[Dict[str, Any]], *, selection: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    avg_confidences = [float(row["avg_confidence"]) for row in ok_rows]
    avg_strengths = [float(row["avg_preference_strength"]) for row in ok_rows]
    n_ok = max(1, len(ok_rows))
    exclusion_summary = {}
    for tag in EXCLUSION_TAGS:
        exclusion_summary[f"fraction_{tag}_pass1"] = float(
            sum(tag in row["pass1"]["judgment"].get("exclusion_tags", []) for row in ok_rows) / n_ok
        )
        exclusion_summary[f"fraction_{tag}_pass2"] = float(
            sum(tag in row["pass2"]["judgment"].get("exclusion_tags", []) for row in ok_rows) / n_ok
        )
        exclusion_summary[f"fraction_{tag}_either_pass"] = float(
            sum(
                (tag in row["pass1"]["judgment"].get("exclusion_tags", []))
                or (tag in row["pass2"]["judgment"].get("exclusion_tags", []))
                for row in ok_rows
            )
            / n_ok
        )
    return {
        "count": len(rows),
        "selection": selection,
        "config": config,
        "summary": {
            "success_count": len(ok_rows),
            "error_count": len(error_rows),
            "fraction_best_agree": float(sum(bool(row["best_agree"]) for row in ok_rows) / n_ok),
            "fraction_worst_agree": float(sum(bool(row["worst_agree"]) for row in ok_rows) / n_ok),
            "fraction_pair_agree": float(sum(bool(row["pair_agree"]) for row in ok_rows) / n_ok),
            "fraction_usable_pass1": float(sum(bool(row["pass1"]["judgment"]["usable_for_training"]) for row in ok_rows) / n_ok),
            "fraction_usable_pass2": float(sum(bool(row["pass2"]["judgment"]["usable_for_training"]) for row in ok_rows) / n_ok),
            "fraction_usable_both_passes": float(sum(bool(row["usable_both_passes"]) for row in ok_rows) / n_ok),
            "mean_avg_confidence": _safe_mean(avg_confidences),
            "mean_avg_preference_strength": _safe_mean(avg_strengths),
            "mean_avg_training_quality": _safe_mean([float(row["avg_training_quality"]) for row in ok_rows]),
            "mean_kept_candidates": _safe_mean([float(row["num_kept_candidates"]) for row in ok_rows]),
            **exclusion_summary,
        },
        "pair_agree_examples": [_compact_row(row) for row in ok_rows if row.get("pair_agree")][:5],
        "pair_disagree_examples": [_compact_row(row) for row in ok_rows if not row.get("pair_agree")][:5],
        "error_examples": error_rows[:5],
    }


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "row_id": row.get("row_id"),
        "score": row.get("score"),
        "prompt_text": row.get("prompt_text"),
        "avg_confidence": row.get("avg_confidence"),
        "avg_preference_strength": row.get("avg_preference_strength"),
        "avg_training_quality": row.get("avg_training_quality"),
        "best_agree": row.get("best_agree"),
        "worst_agree": row.get("worst_agree"),
        "pair_agree": row.get("pair_agree"),
        "usable_both_passes": row.get("usable_both_passes"),
        "pass1": {
            "best_sample_index": row["pass1"]["best_sample_index"],
            "worst_sample_index": row["pass1"]["worst_sample_index"],
            "short_reason": row["pass1"]["judgment"]["short_reason"],
            "usable_for_training": row["pass1"]["judgment"]["usable_for_training"],
            "usable_reason": row["pass1"]["judgment"]["usable_reason"],
            "exclusion_tags": row["pass1"]["judgment"].get("exclusion_tags", []),
        },
        "pass2": {
            "best_sample_index": row["pass2"]["best_sample_index"],
            "worst_sample_index": row["pass2"]["worst_sample_index"],
            "short_reason": row["pass2"]["judgment"]["short_reason"],
            "usable_for_training": row["pass2"]["judgment"]["usable_for_training"],
            "usable_reason": row["pass2"]["judgment"]["usable_reason"],
            "exclusion_tags": row["pass2"]["judgment"].get("exclusion_tags", []),
        },
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.fmean(xs))


if __name__ == "__main__":
    main()
