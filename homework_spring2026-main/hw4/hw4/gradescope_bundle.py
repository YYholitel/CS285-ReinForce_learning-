from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_RUN_NAMES = {
    "format_copy_grpo",
    "format_copy_reinforce",
    "math_hard_grpo",
    "math_hard_reinforce",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a compact Gradescope submission bundle from run directories.")
    ap.add_argument(
        "--run_dir",
        action="append",
        required=True,
        help=(
            "Path to a completed training run output directory. Pass once per run you want to include. "
            "Partial bundles are allowed; missing required runs will simply remain ungraded / score zero "
            "until you add them in a later submission."
        ),
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="submissions/hw4_gradescope_submission",
        help="Directory where the compact bundle folder and zip should be written.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory / zip if present.",
    )
    return ap.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _run_name_from_config(config: dict[str, Any]) -> str:
    task = str(config.get("task", "")).strip()
    algo = str(config.get("algo", "")).strip()
    if not task or not algo:
        raise ValueError("config.json must include non-empty 'task' and 'algo' fields.")
    return f"{task}_{algo}"


def _find_latest_checkpoint_dir(run_dir: Path) -> Path:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"Missing checkpoints directory: {ckpt_root}")

    def _step_num(path: Path) -> int:
        try:
            return int(path.name.split("_", 1)[1])
        except Exception as e:
            raise ValueError(f"Unexpected checkpoint directory name: {path.name}") from e

    candidates = [path for path in ckpt_root.glob("step_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No step_* checkpoint directories found under {ckpt_root}")
    return max(candidates, key=_step_num)


def _write_zip_from_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = Path(source_dir.name) / path.relative_to(source_dir)
            zf.write(path, arcname=str(arcname))


def build_bundle(run_dirs: list[Path], output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output zip already exists: {zip_path}")
        zip_path.unlink()

    manifest: dict[str, Any] = {
        "runs": {},
        "required_runs_expected": sorted(EXPECTED_RUN_NAMES),
    }
    seen_run_names: set[str] = set()

    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.json in run directory: {run_dir}")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics.jsonl in run directory: {run_dir}")

        config = _load_json(config_path)
        run_name = _run_name_from_config(config)
        if run_name not in EXPECTED_RUN_NAMES:
            raise ValueError(
                f"Run directory {run_dir} has unexpected task/algo combination '{run_name}'. "
                f"Expected one of: {sorted(EXPECTED_RUN_NAMES)}"
            )
        if run_name in seen_run_names:
            raise ValueError(f"Duplicate run for task/algo combination '{run_name}'")
        seen_run_names.add(run_name)

        latest_ckpt_dir = _find_latest_checkpoint_dir(run_dir)
        checkpoint_meta_path = latest_ckpt_dir / "meta.json"
        adapter_manifest_path = latest_ckpt_dir / "adapter_manifest.json"
        if not checkpoint_meta_path.is_file():
            raise FileNotFoundError(f"Missing final checkpoint meta.json: {checkpoint_meta_path}")
        if not adapter_manifest_path.is_file():
            raise FileNotFoundError(f"Missing adapter_manifest.json: {adapter_manifest_path}")

        target_dir = output_dir / run_name
        _copy_file(config_path, target_dir / "config.json")
        _copy_file(metrics_path, target_dir / "metrics.jsonl")
        _copy_file(checkpoint_meta_path, target_dir / "latest_checkpoint" / "meta.json")
        _copy_file(adapter_manifest_path, target_dir / "latest_checkpoint" / "adapter_manifest.json")

        checkpoint_meta = _load_json(checkpoint_meta_path)
        adapter_manifest = _load_json(adapter_manifest_path)
        manifest["runs"][run_name] = {
            "source_run_dir": str(run_dir),
            "latest_checkpoint_dir": str(latest_ckpt_dir),
            "checkpoint_step": checkpoint_meta.get("step"),
            "task": checkpoint_meta.get("task"),
            "algo": checkpoint_meta.get("algo"),
            "adapter_file_count": adapter_manifest.get("adapter_file_count"),
            "adapter_total_bytes": adapter_manifest.get("adapter_total_bytes"),
        }

    missing = sorted(EXPECTED_RUN_NAMES - seen_run_names)
    manifest["included_runs"] = sorted(seen_run_names)
    manifest["missing_required_runs"] = missing

    (output_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_zip_from_dir(output_dir, zip_path)
    return zip_path


def main() -> None:
    args = _parse_args()
    run_dirs = [Path(p) for p in args.run_dir]
    zip_path = build_bundle(
        run_dirs=run_dirs,
        output_dir=Path(args.output_dir),
        overwrite=bool(args.overwrite),
    )
    included = sorted({_run_name_from_config(_load_json(path / "config.json")) for path in run_dirs})
    missing = sorted(EXPECTED_RUN_NAMES - set(included))
    print(f"Wrote Gradescope bundle directory: {Path(args.output_dir).resolve()}")
    print(f"Wrote Gradescope bundle zip: {zip_path.resolve()}")
    print(f"Included runs: {included}")
    if missing:
        print(f"Missing required runs not yet included: {missing}")


if __name__ == "__main__":
    main()
