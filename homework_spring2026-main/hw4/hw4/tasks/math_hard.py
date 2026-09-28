from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import get_dataset_config_names, load_dataset

from hw4.tasks.base import Task, TaskExample
from hw4.utils.answer_parsing import (
    extract_last_boxed_content,
    extract_last_number,
    extract_number_from_boxed_answer,
    parse_number,
)

LEVEL_RE = re.compile(r"(\d+)")
MATH_DATASET_ID_DEFAULT = "the-jb/hendrycks-math"


def _parse_level(level_text: Any) -> Optional[int]:
    m = LEVEL_RE.search(str(level_text))
    if m is None:
        return None
    return int(m.group(1))


class MathHardTask(Task):
    """Hard MATH subset (default: level-5) with deterministic reduced test-subset eval."""

    name = "math_hard"

    def __init__(
        self,
        split_train: str = "train",
        split_test: str = "test",
        seed: int = 0,
        train_levels: Sequence[int] = (5,),
        eval_subset_size: int = 512,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
        shaped_reward: float = 0.0,
        relaxed_shaping_weight: float = 0.0,
        relaxed_correct_bonus: float = 0.1,
        use_fallback_number: bool = True,
    ):
        self.rng = random.Random(seed)
        self.levels = tuple(sorted({int(x) for x in train_levels}))
        self.correct_reward = float(correct_reward)
        self.format_reward = float(format_reward)
        self.shaped_reward = float(shaped_reward)
        self.relaxed_shaping_weight = float(relaxed_shaping_weight)
        self.relaxed_correct_bonus = float(relaxed_correct_bonus)
        self.use_fallback_number = bool(use_fallback_number)
        self.eval_subset_size = int(max(1, eval_subset_size))

        dataset_source_id, ds = self._load_math_dataset(MATH_DATASET_ID_DEFAULT)
        self.dataset_source_hf_id = dataset_source_id
        train_split_name, train_split_ds, test_split_name, test_split_ds = self._resolve_splits(
            ds,
            split_train=split_train,
            split_test=split_test,
        )
        train_rows_all = self._build_rows(
            train_split_ds,
            allowed_levels=self.levels,
            split_name=train_split_name,
        )
        if len(train_rows_all) < 3:
            raise RuntimeError(
                "math_hard had too few usable training rows after filtering. "
                f"usable_train_rows={len(train_rows_all)} levels={self.levels} "
                f"dataset_source={dataset_source_id}"
            )
        if test_split_ds is None:
            # Fallback only if the dataset unexpectedly has no explicit test split.
            holdout_rng = random.Random(seed + 22222)
            holdout_rng.shuffle(train_rows_all)
            n_test_holdout = max(64, int(0.1 * len(train_rows_all)))
            n_test_holdout = min(n_test_holdout, max(1, len(train_rows_all) - 2))
            test_rows = list(train_rows_all[:n_test_holdout])
            train_rows = list(train_rows_all[n_test_holdout:])
            test_split_name = f"{train_split_name}_heldout_test"
        else:
            train_rows = train_rows_all
            test_rows = self._build_rows(
                test_split_ds,
                allowed_levels=self.levels,
                split_name=test_split_name,
            )
        if len(train_rows) < 2:
            raise RuntimeError(
                "math_hard training rows became too small after constructing held-out test set. "
                f"usable_train_rows={len(train_rows)} dataset_source={dataset_source_id}"
            )

        self.train_rows = list(train_rows)
        if not self.train_rows:
            raise RuntimeError("math_hard train_rows is empty after filtering.")
        self.test_rows = test_rows
        self.test_eval_subset_rows = list(self.test_rows[: min(self.eval_subset_size, len(self.test_rows))])
        if not self.test_eval_subset_rows:
            raise RuntimeError("math_hard deterministic test-eval subset is empty after filtering.")

        self.dataset_train_split_name = train_split_name
        self.dataset_test_split_name = test_split_name
        has_explicit_test = float(test_split_ds is not None)
        self.dataset_stats: Dict[str, float] = {
            "math_hard/dataset_total_rows_source_train_after_filtering_numeric_and_level": float(len(train_rows_all)),
            "math_hard/dataset_total_rows_train_after_filtering_numeric_and_level": float(len(train_rows)),
            "math_hard/dataset_total_rows_train_used_for_rl_sampling": float(len(self.train_rows)),
            "math_hard/dataset_total_rows_test_after_filtering_numeric_and_level": float(len(self.test_rows)),
            "math_hard/dataset_total_rows_deterministic_reduced_test_subset_used_by_default_for_eval": float(
                len(self.test_eval_subset_rows)
            ),
            "math_hard/dataset_source_has_explicit_test_split_indicator": has_explicit_test,
            "math_hard/dataset_levels_included_max_difficulty_subset_indicator_level_5": float(5 in self.levels),
        }

    @staticmethod
    def _load_math_dataset(dataset_id: str):
        try:
            return dataset_id, load_dataset(dataset_id)
        except Exception as e:
            msg = str(e)
            if "Config name is missing" not in msg and "BuilderConfig" not in msg:
                raise RuntimeError(f"Failed to load dataset_id={dataset_id}: {type(e).__name__}: {e}") from e
            try:
                cfgs = list(get_dataset_config_names(dataset_id))
            except Exception as e_cfg:
                raise RuntimeError(
                    f"Failed to load dataset_id={dataset_id}. Config lookup error: {type(e_cfg).__name__}: {e_cfg}"
                ) from e_cfg
            for cfg in cfgs:
                try:
                    return f"{dataset_id}:{cfg}", load_dataset(dataset_id, cfg)
                except Exception:
                    continue
            raise RuntimeError(
                f"Failed to load dataset_id={dataset_id}. Dataset requires config and none worked from {cfgs}."
            )

    @staticmethod
    def _resolve_splits(ds, split_train: str, split_test: str):
        if not hasattr(ds, "keys"):
            return split_train, ds, split_test, None
        split_names = list(ds.keys())
        if not split_names:
            raise RuntimeError("Loaded dataset has no splits.")

        train_name = split_train if split_train in ds else None
        if train_name is None:
            for c in ("train", "training"):
                if c in ds:
                    train_name = c
                    break
        if train_name is None:
            train_name = split_names[0]
        train_ds = ds[train_name]

        test_name = split_test if (split_test in ds and split_test != train_name) else None
        if test_name is None:
            for c in ("test", "validation", "val", "dev"):
                if c in ds and c != train_name:
                    test_name = c
                    break
        if test_name is None:
            return train_name, train_ds, "none", None
        return train_name, train_ds, test_name, ds[test_name]

    @staticmethod
    def _build_rows(split, allowed_levels: Sequence[int], split_name: str) -> List[Dict[str, Any]]:
        allowed_set = set(int(x) for x in allowed_levels)
        rows: List[Dict[str, Any]] = []
        for i, ex in enumerate(split):
            level_raw = ex.get("level", ex.get("difficulty"))
            level = _parse_level(level_raw)
            if level is None:
                continue
            if allowed_set and level not in allowed_set:
                continue
            problem_raw = ex.get("problem", ex.get("question", ex.get("prompt", "")))
            problem = str(problem_raw).strip()
            solution_raw = ex.get("solution", ex.get("answer", ex.get("final_answer", "")))
            solution = str(solution_raw).strip()
            if not problem or not solution:
                continue
            gt = extract_number_from_boxed_answer(solution)
            if gt is None:
                gt = parse_number(solution)
            if gt is None or (not math.isfinite(float(gt))):
                continue
            rows.append(
                {
                    "row_idx_in_split": int(i),
                    "split_name": split_name,
                    "problem": problem,
                    "solution": solution,
                    "subject": str(ex.get("type", ex.get("subject", ex.get("category", "unknown")))),
                    "level": int(level),
                    "gt": float(gt),
                    "gt_source": "boxed_or_direct_number",
                }
            )
        return rows

    def _build_messages(self, problem: str) -> List[Dict[str, str]]:
        system = (
            "Solve the competition-level math problem.\n"
            "Return the final answer in this exact format: \\boxed{NUMBER}\n"
            "Do not output XML, ####, or extra prose.\n"
            "Output only the boxed final answer."
        )
        user = problem + "\n\nReturn only: \\boxed{NUMBER}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def sample_train_batch(self, batch_size: int) -> List[TaskExample]:
        if not self.train_rows:
            raise RuntimeError("math_hard training split is empty after filtering.")
        out: List[TaskExample] = []
        for _ in range(batch_size):
            row = self.train_rows[self.rng.randrange(len(self.train_rows))]
            out.append(
                TaskExample(
                    meta={
                        "split": "train",
                        "row_idx_in_original_split": int(row["row_idx_in_split"]),
                        "subject": row["subject"],
                        "level": int(row["level"]),
                        "gt": float(row["gt"]),
                        "question": row["problem"],
                        "gt_source": row["gt_source"],
                    },
                    messages=self._build_messages(row["problem"]),
                    task_name=self.name,
                )
            )
        return out

    def _shaping(self, pred: Optional[float], gt: float) -> float:
        if pred is None:
            return 0.0
        rel_err = abs(pred - gt) / max(1.0, abs(gt))
        return self.shaped_reward * math.exp(-4.0 * rel_err)

    def reward(self, example: TaskExample, completion_text: str) -> Tuple[float, Dict[str, Any]]:
        gt = float(example.meta["gt"])
        pred_boxed = extract_number_from_boxed_answer(completion_text)
        boxed_content = extract_last_boxed_content(completion_text)
        used_fallback = False
        pred_fallback = None
        if pred_boxed is None and self.use_fallback_number:
            pred_fallback = extract_last_number(completion_text)
            used_fallback = pred_fallback is not None

        has_boxed = boxed_content is not None
        pred_relaxed = pred_boxed if pred_boxed is not None else pred_fallback
        exact_boxed = (pred_boxed is not None) and (abs(pred_boxed - gt) < 1e-6)
        exact_relaxed = (pred_relaxed is not None) and (abs(pred_relaxed - gt) < 1e-6)
        exact_relaxed_no_boxed = exact_relaxed and (not exact_boxed)

        text_l = completion_text.lower()
        has_boxed_keyword = "\\boxed{" in text_l

        use_boxed_shaping = self.shaped_reward > 0.0
        use_relaxed_shaping = use_boxed_shaping and (self.relaxed_shaping_weight > 0.0)
        reward_format = self.format_reward if has_boxed_keyword else 0.0
        reward_shaping_boxed = self._shaping(pred_boxed, gt) if use_boxed_shaping else 0.0
        reward_shaping_relaxed = self._shaping(pred_relaxed, gt) if use_relaxed_shaping else 0.0
        reward_exact = self.correct_reward if exact_boxed else 0.0

        reward = 0.0
        reward += reward_format
        if use_boxed_shaping:
            reward += reward_shaping_boxed
        if use_relaxed_shaping:
            reward += self.relaxed_shaping_weight * reward_shaping_relaxed
        if exact_relaxed_no_boxed:
            reward += self.relaxed_correct_bonus * self.correct_reward
        reward += reward_exact

        info = {
            "math_hard/is_exact_match_using_number_parsed_from_boxed_answer": float(exact_boxed),
            "math_hard/is_exact_match_using_relaxed_last_number_parser": float(exact_relaxed),
            "math_hard/is_exact_match_only_with_relaxed_parser_not_boxed": float(exact_relaxed_no_boxed),
            "math_hard/completion_contains_boxed_answer_pattern": float(has_boxed),
            "math_hard/completion_contains_literal_backslash_boxed_open_brace": float(has_boxed_keyword),
            "math_hard/used_relaxed_fallback_last_number_parser": float(used_fallback),
            "math_hard/reward_component_format_contains_boxed_pattern": float(reward_format),
            "math_hard/reward_component_relaxed_exact_match_bonus_added_to_total_reward": float(
                self.relaxed_correct_bonus * self.correct_reward if exact_relaxed_no_boxed else 0.0
            ),
            "math_hard/reward_component_exact_match_from_boxed_prediction": float(reward_exact),
            "math_hard/predicted_number_from_boxed_answer_parser": float(pred_boxed) if pred_boxed is not None else None,
            "math_hard/predicted_number_from_relaxed_last_number_parser": float(pred_fallback)
            if pred_fallback is not None
            else None,
            "math_hard/ground_truth_number": float(gt),
            "math_hard/problem_level_integer": float(example.meta.get("level", 0)),
        }
        if use_boxed_shaping:
            info["math_hard/reward_component_numeric_shaping_from_boxed_prediction"] = float(reward_shaping_boxed)
        if use_relaxed_shaping:
            info["math_hard/reward_component_numeric_shaping_from_relaxed_prediction"] = float(reward_shaping_relaxed)
            info["math_hard/reward_component_weighted_relaxed_shaping_term_added_to_total_reward"] = float(
                self.relaxed_shaping_weight * reward_shaping_relaxed
            )
        return float(reward), info

    def _get_eval_pool(self, split: str) -> List[Dict[str, Any]]:
        if split in ("test_subset", "test"):
            return self.test_eval_subset_rows
        if split in ("test_full", "full_test"):
            return self.test_rows
        raise ValueError(f"Unsupported split for math_hard.evaluate: {split}")

    def evaluate(
        self,
        generate_fn,
        max_new_tokens: int = 512,
        limit: Optional[int] = 512,
        split: str = "test_subset",
        generate_batch_fn=None,
        eval_batch_size: int = 1,
    ) -> Dict[str, float]:
        if eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}")
        pool = self._get_eval_pool(split)
        if not pool:
            raise RuntimeError(f"math_hard {split} split is empty after filtering.")
        n = len(pool) if limit is None else min(limit, len(pool))

        exact = 0
        exact_relaxed = 0
        exact_relaxed_no_boxed = 0
        has_boxed = 0
        fallback_used = 0

        def _accumulate_metrics(row: Dict[str, Any], completion: str) -> None:
            nonlocal exact, exact_relaxed, exact_relaxed_no_boxed, has_boxed, fallback_used
            gt = float(row["gt"])
            pred_boxed = extract_number_from_boxed_answer(completion)
            if extract_last_boxed_content(completion) is not None:
                has_boxed += 1
            if pred_boxed is not None and abs(pred_boxed - gt) < 1e-6:
                exact += 1
                exact_relaxed += 1
                return

            pred_relaxed = extract_last_number(completion)
            if pred_relaxed is not None:
                fallback_used += 1
                if abs(pred_relaxed - gt) < 1e-6:
                    exact_relaxed += 1
                    if pred_boxed is None:
                        exact_relaxed_no_boxed += 1

        eval_rows = pool[:n]
        if generate_batch_fn is None:
            for row in eval_rows:
                completion = generate_fn(self._build_messages(row["problem"]), max_new_tokens=max_new_tokens)
                _accumulate_metrics(row, completion)
        else:
            batch_size = int(max(1, eval_batch_size))
            for start in range(0, n, batch_size):
                batch_rows = eval_rows[start : start + batch_size]
                messages_batch = [self._build_messages(row["problem"]) for row in batch_rows]
                completions = generate_batch_fn(messages_batch, max_new_tokens=max_new_tokens)
                if completions is None or len(completions) != len(batch_rows):
                    raise RuntimeError(
                        "generate_batch_fn must return one completion per prompt. "
                        f"got={0 if completions is None else len(completions)} "
                        f"expected={len(batch_rows)}"
                    )
                for row, completion in zip(batch_rows, completions):
                    _accumulate_metrics(row, completion)

        exact_boxed_rate = exact / max(1, n)
        exact_relaxed_rate = exact_relaxed / max(1, n)
        prefix = f"eval/math_hard_{split}_split_"
        return {
            prefix + "fraction_exact_match_using_boxed_answer_parser": exact_boxed_rate,
            prefix + "fraction_exact_match_using_relaxed_last_number_parser": exact_relaxed_rate,
            prefix + "fraction_exact_match_only_with_relaxed_parser_not_boxed": exact_relaxed_no_boxed / max(1, n),
            prefix + "gap_relaxed_exact_rate_minus_boxed_exact_rate": exact_relaxed_rate - exact_boxed_rate,
            prefix + "fraction_completions_containing_boxed_answer_pattern": has_boxed / max(1, n),
            prefix + "fraction_completions_where_relaxed_fallback_parser_was_used": fallback_used / max(1, n),
            prefix + "number_of_evaluation_examples": float(n),
            prefix + "dataset_pool_size_available_for_this_split_after_filtering": float(len(pool)),
        }
