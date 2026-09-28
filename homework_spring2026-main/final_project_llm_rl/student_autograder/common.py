from __future__ import annotations

import concurrent.futures
import json
import os
import random
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import certifi

DATA_DIR = Path(__file__).resolve().parent.parent / "public_eval"
DEFAULT_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_JUDGE_MODEL = "gpt-5.4"
EXCLUSION_TAGS = [
    "explicit_sexual_content",
    "medical_mental_health_or_crisis",
    "legal_or_financial_advice",
    "copyright_or_fanfic",
    "garbled_or_underspecified",
    "high_factual_uncertainty",
    "all_candidates_weak",
]
PAIR_LABELS = ("A", "B")

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
    api_key: str
    judge_model: str
    api_url: str = DEFAULT_API_URL
    reasoning_effort: str = "none"
    verbosity: str = "low"
    max_output_tokens: int = 220
    timeout_seconds: float = 120.0
    max_retries: int = 5
    max_workers: int = 8


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_results_json(path: Path, tests: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "score": float(sum(float(test.get("score", 0.0)) for test in tests)),
        "tests": list(tests),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_submission_root(root: Path) -> Path:
    preferred = list(root.rglob("llm_rl_final_proj_public_submission"))
    if preferred:
        preferred.sort(key=lambda p: (len(p.parts), str(p)))
        return preferred[0]
    if (root / "policy_generations").is_dir() or (root / "reward_model").is_dir() or (root / "part2").is_dir():
        return root
    for path in sorted(root.rglob("policy_generations")):
        return path.parent
    raise FileNotFoundError("Could not find llm_rl_final_proj_public_submission in submission.")


def load_public_data() -> Dict[str, Any]:
    return {
        "part1_prompts": load_jsonl(DATA_DIR / "public_test_gen_prompts_128.jsonl"),
        "part1_base": load_jsonl(DATA_DIR / "public_test_gen_base_responses_128.jsonl"),
        "part2_prompts": load_jsonl(DATA_DIR / "public_test_gen_prompts_128.jsonl"),
        "part2_base": load_jsonl(DATA_DIR / "public_test_gen_base_responses_128.jsonl"),
        "reward_prefs": load_jsonl(DATA_DIR / "public_test_prefs_256.jsonl"),
    }


def _normalize_reasoning_effort(model: str, effort: str) -> str:
    model_l = model.lower()
    if effort == "none" and model_l.startswith("gpt-5-mini"):
        return "minimal"
    return effort


def _post_json(*, url: str, api_key: str, payload: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=timeout_seconds, context=ssl_context) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {exc}") from exc


def _post_with_retries(cfg: JudgeConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
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
            time.sleep(min(30.0, 1.5 ** attempt) + random.random() * 0.25)


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
        raise RuntimeError(f"No output_text content found in response: {json.dumps(response_json)[:800]}")
    return text


def _build_pair_input(prompt_text: str, first_label: str, first_text: str, second_label: str, second_text: str) -> str:
    return "\n".join(
        [
            "User prompt:",
            prompt_text.strip(),
            "",
            "Candidate responses:",
            f"[{first_label}] (sample_index=0)",
            first_text.strip(),
            "",
            f"[{second_label}] (sample_index=1)",
            second_text.strip(),
            "",
            "Return only the JSON object that matches the schema.",
        ]
    )


def _normalize_pair_judgment(judgment: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(judgment)
    raw_tags = out.get("exclusion_tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError("Judge returned a non-list exclusion_tags field.")
    deduped: List[str] = []
    seen = set()
    for tag in raw_tags:
        if tag in seen:
            continue
        deduped.append(tag)
        seen.add(tag)
    out["exclusion_tags"] = deduped
    if deduped:
        out["usable_for_training"] = False
        out["training_quality"] = min(int(out["training_quality"]), 2)
    if out.get("best_label") == out.get("worst_label"):
        better = str(out.get("best_label", "A"))
        out["worst_label"] = "B" if better == "A" else "A"
        if "all_candidates_weak" not in deduped:
            out["exclusion_tags"] = deduped + ["all_candidates_weak"]
        out["usable_for_training"] = False
        out["training_quality"] = min(int(out["training_quality"]), 1)
        short_reason = str(out.get("short_reason", "")).strip()
        usable_reason = str(out.get("usable_reason", "")).strip()
        out["short_reason"] = (
            short_reason + " " if short_reason else ""
        ) + "The judge collapsed best and worst onto the same candidate, so this row was coerced into an excluded weak-example comparison."
        out["usable_reason"] = (
            usable_reason + " " if usable_reason else ""
        ) + "This row was excluded because the judge did not identify a stable distinct best and worst candidate."
    return out


def _judge_once(prompt_text: str, first_label: str, first_text: str, second_label: str, second_text: str, cfg: JudgeConfig) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "best_label": {"type": "string", "enum": list(PAIR_LABELS)},
            "worst_label": {"type": "string", "enum": list(PAIR_LABELS)},
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
    payload = {
        "model": cfg.judge_model,
        "store": False,
        "instructions": JUDGE_INSTRUCTIONS,
        "input": _build_pair_input(prompt_text, first_label, first_text, second_label, second_text),
        "reasoning": {"effort": _normalize_reasoning_effort(cfg.judge_model, cfg.reasoning_effort)},
        "text": {
            "verbosity": cfg.verbosity,
            "format": {
                "type": "json_schema",
                "name": "pairwise_grade",
                "strict": True,
                "schema": schema,
            },
        },
        "max_output_tokens": cfg.max_output_tokens,
    }
    response = _post_with_retries(cfg, payload)
    judgment = json.loads(_extract_output_text(response))
    judgment = _normalize_pair_judgment(judgment)
    if judgment["best_label"] == judgment["worst_label"]:
        raise ValueError("Judge returned the same label for best and worst.")
    invalid_tags = [tag for tag in judgment.get("exclusion_tags", []) if tag not in EXCLUSION_TAGS]
    if invalid_tags:
        raise ValueError(f"Unknown exclusion tags: {invalid_tags}")
    return judgment


def _map_winner(judgment: Mapping[str, Any], first_owner: str, second_owner: str) -> str:
    label = str(judgment["best_label"])
    return first_owner if label == "A" else second_owner


def grade_policy_submission(
    prompt_rows: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
    student_rows: Sequence[Mapping[str, Any]],
    judge_cfg: JudgeConfig,
) -> Dict[str, Any]:
    prompts_by_id = {str(r["row_id"]): r for r in prompt_rows}
    base_by_id = {str(r["row_id"]): r for r in base_rows}
    student_by_id = {str(r.get("row_id")): r for r in student_rows}
    missing = sorted(set(prompts_by_id) - set(student_by_id))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} required policy rows; first few: {missing[:5]}")

    def judge_row(row_id: str) -> Dict[str, Any]:
        prompt = str(prompts_by_id[row_id].get("prompt_text", ""))
        student_text = str(student_by_id[row_id].get("response_text", "")).strip()
        base_text = str(base_by_id[row_id].get("response_text", "")).strip()
        if not student_text:
            raise RuntimeError(f"Empty response_text for row_id={row_id}")
        if not base_text:
            raise RuntimeError(f"Empty cached base response for row_id={row_id}")
        pass1 = _judge_once(prompt, "A", student_text, "B", base_text, judge_cfg)
        pass2 = _judge_once(prompt, "A", base_text, "B", student_text, judge_cfg)
        winner1 = _map_winner(pass1, "student", "base")
        winner2 = _map_winner(pass2, "base", "student")
        loser1 = "base" if winner1 == "student" else "student"
        loser2 = "base" if winner2 == "student" else "student"
        pair_agree = winner1 == winner2 and loser1 == loser2
        usable_both = bool(pass1.get("usable_for_training") and pass2.get("usable_for_training"))
        return {
            "row_id": row_id,
            "status": "ok",
            "pass1": pass1,
            "pass2": pass2,
            "winner": winner1 if pair_agree else None,
            "pair_agree": pair_agree,
            "usable_both_passes": usable_both,
        }

    ordered_ids = [str(r["row_id"]) for r in prompt_rows]
    rows: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=judge_cfg.max_workers) as pool:
        futures = {pool.submit(judge_row, row_id): row_id for row_id in ordered_ids}
        for fut in concurrent.futures.as_completed(futures):
            row_id = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                rows.append({"row_id": row_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    rows.sort(key=lambda r: ordered_ids.index(str(r["row_id"])))

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    usable = [r for r in ok_rows if r.get("pair_agree") and r.get("usable_both_passes")]
    student_wins = sum(r.get("winner") == "student" for r in usable)
    base_wins = sum(r.get("winner") == "base" for r in usable)
    rate = float(student_wins / len(usable)) if usable else 0.0
    return {
        "count_total_rows": len(rows),
        "count_ok_rows": len(ok_rows),
        "count_pair_agree_usable_rows": len(usable),
        "count_student_wins_pair_agree_usable": student_wins,
        "count_base_wins_pair_agree_usable": base_wins,
        "policy_win_rate_pair_agree_usable": rate,
        "error_examples": [r for r in rows if r.get("status") == "error"][:5],
    }


def grade_reward_model_submission(submitted_rows: Sequence[Mapping[str, Any]], prefs_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prefs_by_id = {str(r["row_id"]): r for r in prefs_rows}
    sub_by_id = {str(r.get("row_id")): r for r in submitted_rows}
    missing = sorted(set(prefs_by_id) - set(sub_by_id))
    extra = sorted(set(sub_by_id) - set(prefs_by_id))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} required reward-model rows; first few: {missing[:5]}")
    if extra:
        raise RuntimeError(f"Found {len(extra)} unexpected reward-model rows; first few: {extra[:5]}")
    correct = 0
    total = 0
    for row_id in prefs_by_id:
        row = sub_by_id[row_id]
        chosen = float(row["chosen_score"])
        rejected = float(row["rejected_score"])
        total += 1
        if chosen > rejected:
            correct += 1
    return {
        "count_total_rows": total,
        "count_correct_rows": correct,
        "pair_accuracy": float(correct / total) if total else 0.0,
    }
