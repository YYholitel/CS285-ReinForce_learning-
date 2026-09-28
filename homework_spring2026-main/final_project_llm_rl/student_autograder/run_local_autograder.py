from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from common import (
    DEFAULT_JUDGE_MODEL,
    JudgeConfig,
    grade_policy_submission,
    grade_reward_model_submission,
    load_jsonl,
    load_public_data,
    resolve_submission_root,
    write_results_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the local LLM-evaluation autograder on a submission directory.")
    ap.add_argument(
        "--submission_dir",
        type=Path,
        default=Path("llm_rl_final_proj_public_submission"),
        help="Path to the submission directory or to a larger folder containing it.",
    )
    ap.add_argument(
        "--output_json",
        type=Path,
        default=Path("student_autograder_results.json"),
        help="Where to write the JSON results summary.",
    )
    return ap.parse_args()


def _grade_tests(root: Path, judge_cfg: JudgeConfig, thresholds: dict, public: dict) -> list[dict]:
    tests: list[dict] = []

    def add_test(name: str, passed: bool, max_score: float, output: str) -> None:
        tests.append(
            {
                "name": name,
                "score": float(max_score if passed else 0.0),
                "max_score": float(max_score),
                "status": "passed" if passed else "failed",
                "output": output,
                "visibility": "visible",
            }
        )

    def grade_policy_test(name: str) -> dict:
        path = root / "policy_generations" / f"{name}.jsonl"
        if not path.is_file():
            return {
                "name": name,
                "score": 0.0,
                "max_score": 1.0,
                "status": "failed",
                "output": f"Missing policy_generations/{name}.jsonl",
                "visibility": "visible",
            }
        metrics = grade_policy_submission(public["part1_prompts"], public["part1_base"], load_jsonl(path), judge_cfg)
        threshold = float(thresholds["part1"][name])
        passed = metrics["policy_win_rate_pair_agree_usable"] >= threshold
        output = (
            f"win_rate={metrics['policy_win_rate_pair_agree_usable']:.4f} "
            f"threshold={threshold:.4f} usable={metrics['count_pair_agree_usable_rows']}"
        )
        if metrics.get("error_examples"):
            first_err = str(metrics["error_examples"][0].get("error", ""))
            output += f" errors={len(metrics['error_examples'])} first_error={first_err[:180]}"
        return {
            "name": name,
            "score": float(1.0 if passed else 0.0),
            "max_score": 1.0,
            "status": "passed" if passed else "failed",
            "output": output,
            "visibility": "visible",
        }

    def grade_part2_variant(label: str, rel_path: Path, threshold: float) -> tuple[str, float]:
        metrics = grade_policy_submission(public["part2_prompts"], public["part2_base"], load_jsonl(rel_path), judge_cfg)
        rate = metrics["policy_win_rate_pair_agree_usable"]
        output = f"{label}={rate:.4f} (threshold {threshold:.4f}, usable={metrics['count_pair_agree_usable_rows']})"
        if metrics.get("error_examples"):
            first_err = str(metrics["error_examples"][0].get("error", ""))
            output += f", errors={len(metrics['error_examples'])}, first_error={first_err[:180]}"
        return output, rate

    reward_path = root / "reward_model" / "public_test_pref_scores.jsonl"
    if not reward_path.is_file():
        add_test("reward_model", False, 1.0, "Missing reward_model/public_test_pref_scores.jsonl")
    else:
        rm_metrics = grade_reward_model_submission(load_jsonl(reward_path), public["reward_prefs"])
        threshold = float(thresholds["part1"]["reward_model_pair_accuracy"])
        add_test(
            "reward_model",
            rm_metrics["pair_accuracy"] >= threshold,
            1.0,
            f"pair_accuracy={rm_metrics['pair_accuracy']:.4f} threshold={threshold:.4f}",
        )

    algos = ("dpo", "ipo", "aot", "grpo", "drgrpo", "gspo")
    policy_pool_workers = int(os.environ.get("LOCAL_AUTOGRADER_POLICY_MAX_WORKERS", "4"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(policy_pool_workers, len(algos))) as pool:
        future_to_algo = {pool.submit(grade_policy_test, algo): algo for algo in algos}
        graded = {}
        for fut in concurrent.futures.as_completed(future_to_algo):
            algo = future_to_algo[fut]
            graded[algo] = fut.result()
    for algo in algos:
        tests.append(graded[algo])

    part2_dir = root / "part2"
    offline_path = part2_dir / "offline_best.jsonl"
    online_path = part2_dir / "online_best.jsonl"
    outputs: list[str] = []
    passed = False
    part2_specs = []
    if offline_path.is_file():
        part2_specs.append(("offline", offline_path, float(thresholds["part2"]["offline_policy_win_rate"])))
    if online_path.is_file():
        part2_specs.append(("online", online_path, float(thresholds["part2"]["online_policy_win_rate"])))
    if part2_specs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(policy_pool_workers, len(part2_specs))) as pool:
            future_to_label = {
                pool.submit(grade_part2_variant, label, path, threshold): (label, threshold)
                for label, path, threshold in part2_specs
            }
            part2_results = {}
            for fut in concurrent.futures.as_completed(future_to_label):
                label, threshold = future_to_label[fut]
                output, rate = fut.result()
                part2_results[label] = output
                passed = passed or (rate >= threshold)
        for label, _, _ in part2_specs:
            outputs.append(part2_results[label])
    if not outputs:
        outputs.append("Missing part2/offline_best.jsonl and part2/online_best.jsonl")
    add_test("part2_best", passed, 1.0, "; ".join(outputs))
    return tests


def main() -> None:
    args = parse_args()
    thresholds = json.loads((Path(__file__).resolve().parent / "thresholds.json").read_text(encoding="utf-8"))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to run the local autograder.")
    judge_model = os.environ.get("LOCAL_AUTOGRADER_JUDGE_MODEL", thresholds.get("judge_model", DEFAULT_JUDGE_MODEL))
    public = load_public_data()
    root = resolve_submission_root(args.submission_dir)
    judge_cfg = JudgeConfig(
        api_key=api_key,
        judge_model=judge_model,
        reasoning_effort=str(thresholds.get("reasoning_effort", "none")),
        max_workers=int(os.environ.get("LOCAL_AUTOGRADER_MAX_WORKERS", "16")),
    )
    tests = _grade_tests(root, judge_cfg, thresholds, public)
    write_results_json(args.output_json, tests)
    total_score = sum(float(test.get("score", 0.0)) for test in tests)
    max_score = sum(float(test.get("max_score", 0.0)) for test in tests)
    print(f"Wrote {args.output_json}  score={total_score:.1f}/{max_score:.1f}")
    for test in tests:
        status = "PASS" if test["status"] == "passed" else "FAIL"
        print(f"[{status}] {test['name']}: {test['output']}")


if __name__ == "__main__":
    main()
