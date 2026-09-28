"""Plot the first-pass CartPole comparison without modifying training code.

Policy-gradient curves use training iteration on the x-axis.  The DQN curve
uses episode.  This is therefore a coarse learning-trend comparison, not a
sample-efficiency comparison; use environment steps for a rigorous comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_DIR = PROJECT_ROOT / "exp"
DEFAULT_DQN_RETURNS = Path(__file__).parent / "dqn_cartpole" / "dqn_returns.npy"
DEFAULT_OUTPUT = Path(__file__).parent / "cartpole_method_comparison.png"


@dataclass(frozen=True)
class PGExperiment:
    names: tuple[str, ...]
    label: str


PG_EXPERIMENTS = (
    PGExperiment(("reinforce", "cartpole"), "REINFORCE"),
    PGExperiment(("reinforce_rtg", "cartpole_rtg"), "REINFORCE + RTG"),
    PGExperiment(
        ("reinforce_rtg_na", "cartpole_rtg_na"),
        "RTG + Advantage Normalization",
    ),
    PGExperiment(
        ("reinforce_baseline", "cartpole_baseline"),
        "RTG + Baseline",
    ),
    PGExperiment(
        ("reinforce_gae", "cartpole_gae"),
        "RTG + Baseline + GAE",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--dqn-returns", type=Path, default=DEFAULT_DQN_RETURNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dqn-window", type=int, default=10)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Plot available curves instead of failing when an experiment is missing.",
    )
    return parser.parse_args()


def experiment_name(run_dir: Path) -> str | None:
    flags_path = run_dir / "flags.json"
    if not flags_path.is_file():
        return None
    try:
        with flags_path.open("r", encoding="utf-8") as file:
            return json.load(file).get("exp_name")
    except (OSError, json.JSONDecodeError):
        return None


def find_latest_log(exp_dir: Path, names: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    if not exp_dir.is_dir():
        return None

    for run_dir in exp_dir.iterdir():
        log_path = run_dir / "log.csv"
        if run_dir.is_dir() and log_path.is_file() and experiment_name(run_dir) in names:
            candidates.append(log_path)

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_pg_returns(log_path: Path) -> np.ndarray:
    with log_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Log has no data rows: {log_path}")

    column = (
        "Eval_AverageReturn"
        if "Eval_AverageReturn" in rows[0]
        else "Train_AverageReturn"
    )
    if column not in rows[0]:
        raise ValueError(f"No average-return column found in {log_path}")
    return np.asarray([float(row[column]) for row in rows], dtype=np.float64)


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window < 1:
        raise ValueError("--dqn-window must be at least 1")
    if values.size < window or window == 1:
        return np.arange(1, values.size + 1), values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    # Each point is positioned at the final episode in its averaging window.
    episodes = np.arange(window, values.size + 1)
    return episodes, smoothed


def main() -> None:
    args = parse_args()
    missing: list[str] = []
    curves: list[tuple[np.ndarray, np.ndarray, str]] = []

    for experiment in PG_EXPERIMENTS:
        log_path = find_latest_log(args.exp_dir, experiment.names)
        if log_path is None:
            missing.append(f"PG: {experiment.names[0]}")
            continue
        returns = read_pg_returns(log_path)
        iterations = np.arange(1, returns.size + 1)
        curves.append((iterations, returns, experiment.label))
        print(f"{experiment.label}: {log_path}")

    if args.dqn_returns.is_file():
        dqn_returns = np.asarray(np.load(args.dqn_returns), dtype=np.float64).reshape(-1)
        if dqn_returns.size == 0:
            raise ValueError(f"DQN returns file is empty: {args.dqn_returns}")
        episodes, dqn_returns = moving_average(dqn_returns, args.dqn_window)
        curves.append((episodes, dqn_returns, f"DQN ({args.dqn_window}-episode MA)"))
        print(f"DQN: {args.dqn_returns}")
    else:
        missing.append(f"DQN: {args.dqn_returns}")

    if missing and not args.allow_missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Cannot create the complete comparison; missing inputs:\n"
            f"  - {formatted}\n"
            "Generate these results first, or pass --allow-missing for a partial plot."
        )
    if not curves:
        raise FileNotFoundError("No experiment data was found to plot.")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for x, y, label in curves:
        linewidth = 2.6 if label.startswith("DQN") else 2.0
        ax.plot(x, y, label=label, linewidth=linewidth)

    ax.set_xlabel("Training Progress (PG: Iteration; DQN: Episode)")
    ax.set_ylabel("Average Return")
    ax.set_title("CartPole: Policy Gradient Variants vs. DQN")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.5,
        -0.18,
        "Coarse trend comparison only: PG and DQN use different x-axis units.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
