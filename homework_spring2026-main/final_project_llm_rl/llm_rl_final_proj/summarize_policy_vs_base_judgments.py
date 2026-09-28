from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize policy-vs-base judged JSONL files.")
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--save_json", type=str, default="")
    return ap.parse_args()


def _load_rows(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.input_jsonl))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    usable_rows = [row for row in ok_rows if bool(row.get("usable_both_passes"))]
    pair_agree_rows = [row for row in usable_rows if bool(row.get("pair_agree"))]

    policy_wins = 0
    base_wins = 0
    ties_or_other = 0
    for row in pair_agree_rows:
        best_idx = int(row["pass1"]["best_sample_index"])
        if best_idx == 0:
            policy_wins += 1
        elif best_idx == 1:
            base_wins += 1
        else:
            ties_or_other += 1

    summary = {
        "count_total_rows": len(rows),
        "count_ok_rows": len(ok_rows),
        "count_usable_both_passes": len(usable_rows),
        "count_pair_agree_usable_rows": len(pair_agree_rows),
        "count_policy_wins_pair_agree_usable": policy_wins,
        "count_base_wins_pair_agree_usable": base_wins,
        "count_ties_or_other_pair_agree_usable": ties_or_other,
        "policy_win_rate_pair_agree_usable": (
            float(policy_wins / max(1, policy_wins + base_wins))
            if (policy_wins + base_wins) > 0
            else 0.0
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
