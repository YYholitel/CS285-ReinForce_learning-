from __future__ import annotations

"""Student extension hook for Part 2 online preference optimization methods.

Suggested use in Part 2:
  - copy patterns from train_rm_grpo.py,
  - define a config/dataclass for your new method,
  - build prompt batches from train_gen,
  - score completions with the reward model,
  - implement your preferred online objective here.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Student extension hook for Part 2 online preference optimization methods."
    )
    parser.parse_args()
    raise NotImplementedError(
        "Implement your method in llm_rl_final_proj/online/train_rm_online_pref.py."
    )


if __name__ == "__main__":
    main()
