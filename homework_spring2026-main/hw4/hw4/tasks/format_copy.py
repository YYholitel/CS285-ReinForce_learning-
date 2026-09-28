from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from hw4.tasks.base import Task, TaskExample
from hw4.utils.answer_parsing import (
    extract_number_from_xml_answer,
    extract_xml_answer_content,
    is_strict_xml_answer,
)


class FormatCopyTask(Task):
    """Easy debugging task for end-to-end RL correctness checks."""

    name = "format_copy"

    def __init__(
        self,
        seed: int = 0,
        min_value: int = -5000,
        max_value: int = 5000,
        correct_reward: float = 1.0,
        format_reward: float = 0.2,
        strict_reward: float = 0.1,
    ):
        self.rng = random.Random(seed)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        self.correct_reward = float(correct_reward)
        self.format_reward = float(format_reward)
        self.strict_reward = float(strict_reward)

    def _sample_target(self) -> int:
        return self.rng.randint(self.min_value, self.max_value)

    def _build_messages(self, target: int) -> List[Dict[str, str]]:
        system = (
            "You are a strict formatter.\n"
            "Return the final answer as XML using exactly one tag: <answer>...</answer>.\n"
            "Output only the XML tag and nothing else."
        )
        user = f"Copy this integer exactly: {target}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def sample_train_batch(self, batch_size: int) -> List[TaskExample]:
        out: List[TaskExample] = []
        for _ in range(batch_size):
            target = self._sample_target()
            out.append(
                TaskExample(
                    meta={"target": target, "split": "train"},
                    messages=self._build_messages(target),
                    task_name=self.name,
                )
            )
        return out

    def reward(self, example: TaskExample, completion_text: str) -> Tuple[float, Dict[str, Any]]:
        target = float(example.meta["target"])
        strict = is_strict_xml_answer(completion_text)
        xml_content = extract_xml_answer_content(completion_text)
        parsed = extract_number_from_xml_answer(completion_text)

        has_xml = xml_content is not None
        exact = (parsed is not None) and (abs(parsed - target) < 1e-6)

        reward = 0.0
        if has_xml:
            reward += self.format_reward
        if strict:
            reward += self.strict_reward
        if exact:
            reward += self.correct_reward

        return float(reward), {
            "format_copy/completion_contains_answer_xml_tag": float(has_xml),
            "format_copy/completion_is_strictly_only_answer_xml_without_extra_text": float(strict),
            "format_copy/predicted_number_matches_target_integer_exactly": float(exact),
            "format_copy/predicted_number_parsed_from_answer_xml": float(parsed) if parsed is not None else None,
            "format_copy/target_integer_ground_truth": float(target),
        }

    def evaluate(
        self,
        generate_fn,
        max_new_tokens: int = 24,
        seed: int = 123,
        n_eval: int = 512,
        generate_batch_fn=None,
        eval_batch_size: int = 1,
    ) -> Dict[str, float]:
        if eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}")
        state = self.rng.getstate()
        self.rng.seed(seed)
        try:
            exact = 0
            has_xml = 0
            strict_xml = 0
            targets = [self._sample_target() for _ in range(n_eval)]

            def _accumulate_metrics(target: int, completion: str) -> None:
                nonlocal exact, has_xml, strict_xml
                if extract_xml_answer_content(completion) is not None:
                    has_xml += 1
                if is_strict_xml_answer(completion):
                    strict_xml += 1
                pred = extract_number_from_xml_answer(completion)
                if pred is not None and abs(pred - target) < 1e-6:
                    exact += 1

            if generate_batch_fn is None:
                for target in targets:
                    completion = generate_fn(self._build_messages(target), max_new_tokens=max_new_tokens)
                    _accumulate_metrics(target, completion)
            else:
                batch_size = int(max(1, eval_batch_size))
                for start in range(0, n_eval, batch_size):
                    batch_targets = targets[start : start + batch_size]
                    messages_batch = [self._build_messages(target) for target in batch_targets]
                    completions: Optional[List[str]] = generate_batch_fn(
                        messages_batch,
                        max_new_tokens=max_new_tokens,
                    )
                    if completions is None or len(completions) != len(batch_targets):
                        raise RuntimeError(
                            "generate_batch_fn must return one completion per prompt. "
                            f"got={0 if completions is None else len(completions)} "
                            f"expected={len(batch_targets)}"
                        )
                    for target, completion in zip(batch_targets, completions):
                        _accumulate_metrics(target, completion)

            return {
                "eval/format_copy_fraction_predicted_number_matches_target_integer_exactly": exact / max(1, n_eval),
                "eval/format_copy_fraction_completions_containing_answer_xml_tag": has_xml / max(1, n_eval),
                "eval/format_copy_fraction_completions_that_are_strict_answer_xml_only": strict_xml / max(1, n_eval),
                "eval/format_copy_number_of_evaluation_examples": float(n_eval),
            }
        finally:
            self.rng.setstate(state)
