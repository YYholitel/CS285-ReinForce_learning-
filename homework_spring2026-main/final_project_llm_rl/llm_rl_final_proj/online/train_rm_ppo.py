from __future__ import annotations

"""Student extension hook for Part 2 PPO-style online RLHF methods, if you want to implement PPO.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Student extension hook for Part 2 PPO-style online RLHF methods.")
    parser.parse_args()
    raise NotImplementedError(
        "Implement your method in llm_rl_final_proj/online/train_rm_ppo.py."
    )


if __name__ == "__main__":
    main()
