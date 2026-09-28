from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "llm-rl-final-project"
VOLUME_NAME = "llm-rl-final-project-volume"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "H100"
INFERENCE_GPU = "A100-40GB"
DEFAULT_CPU = 8.0
DEFAULT_MEMORY_MB = 65536
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24
DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS = 300
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    if not modal.is_local():
        return []
    root = Path(__file__).resolve().parents[1]
    gitignore_path = root / ".gitignore"
    raw_entries = [
        ".git/",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "run_logs/",
        "eval_candidates/",
        "tmp_modal_configs/",
        "tmp_modal_metrics/",
        "tmp_remote_summaries/",
        "llm_rl_final_proj_public_submission/",
        "*.pyc",
        "*.zip",
        ".DS_Store",
    ]
    if gitignore_path.is_file():
        for line in gitignore_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#") or entry.startswith("!"):
                continue
            raw_entries.append(entry)

    patterns: list[str] = []
    for entry in raw_entries:
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.extend([f"{entry}/**", f"**/{entry}/**"])
        elif "/" in entry:
            patterns.extend([entry, f"{entry}/**"])
        else:
            patterns.extend([entry, f"**/{entry}"])
    patterns.extend([".git/**", "**/.git/**"])
    return sorted(set(patterns))


def _to_volume_path(path_value: str) -> str:
    p = Path(path_value)
    if p.is_absolute():
        p_str = str(p)
        if p_str != VOLUME_PATH and not p_str.startswith(f"{VOLUME_PATH}/"):
            print(
                f"[modal][warning] path '{p_str}' is outside '{VOLUME_PATH}'. "
                "Files written there may not persist after the run."
            )
        return path_value
    return str(Path(VOLUME_PATH) / p)


def _rewrite_path_flag(
    args: list[str],
    flag: str,
    *,
    default_relative_if_missing: str | None = None,
    multi_value: bool = False,
) -> list[str]:
    out = list(args)
    found = False
    i = 0
    while i < len(out):
        token = out[i]
        if token == flag:
            found = True
            if i + 1 >= len(out):
                raise ValueError(f"Missing value for {flag}")
            j = i + 1
            while j < len(out):
                if out[j].startswith("--"):
                    break
                out[j] = _to_volume_path(out[j])
                j += 1
                if not multi_value:
                    break
            if j == i + 1:
                raise ValueError(f"Missing value for {flag}")
            i = j
            continue
        if token.startswith(f"{flag}="):
            found = True
            key, value = token.split("=", 1)
            out[i] = f"{key}={_to_volume_path(value)}"
        i += 1
    if not found and default_relative_if_missing is not None:
        out.extend([flag, _to_volume_path(default_relative_if_missing)])
    return out


def _normalize_args(args: tuple[str, ...], *, default_output_dir: str | None = None) -> list[str]:
    normalized = list(args)
    normalized = _rewrite_path_flag(normalized, "--output_dir", default_relative_if_missing=default_output_dir)
    normalized = _rewrite_path_flag(normalized, "--adapter_path")
    normalized = _rewrite_path_flag(normalized, "--reward_adapter_path")
    normalized = _rewrite_path_flag(normalized, "--save_json")
    normalized = _rewrite_path_flag(normalized, "--save_preferences_jsonl")
    normalized = _rewrite_path_flag(normalized, "--save_keep_row_ids_json")
    normalized = _rewrite_path_flag(normalized, "--save_recommended_jsonl")
    normalized = _rewrite_path_flag(normalized, "--summary_json")
    normalized = _rewrite_path_flag(normalized, "--output_jsonl")
    normalized = _rewrite_path_flag(normalized, "--input_jsonl", multi_value=True)
    normalized = _rewrite_path_flag(normalized, "--prompts_jsonl")
    normalized = _rewrite_path_flag(normalized, "--prefs_jsonl")
    normalized = _rewrite_path_flag(normalized, "--test_gen_jsonl")
    normalized = _rewrite_path_flag(normalized, "--test_prefs_jsonl")
    normalized = _rewrite_path_flag(normalized, "--base_candidates_jsonl")
    return normalized


def _is_wandb_enabled(args: tuple[str, ...] | list[str]) -> bool:
    enabled = True
    for token in args:
        if token == "--no-wandb_enabled":
            enabled = False
        elif token == "--wandb_enabled":
            enabled = True
    return enabled


def _assert_wandb_credentials_available_if_needed(args: tuple[str, ...] | list[str]) -> None:
    if not _is_wandb_enabled(args):
        return
    has_netrc = Path("/root/.netrc").is_file()
    has_api_key_env = bool(os.environ.get("WANDB_API_KEY"))
    if not has_netrc and not has_api_key_env:
        raise RuntimeError(
            "W&B logging is enabled for training, but no credentials were found in the Modal container. "
            "Run `uvx wandb login` locally (so ~/.netrc is copied), or export WANDB_API_KEY before modal run, "
            "or pass `--no-wandb_enabled`."
        )


def _run_subprocess_with_periodic_volume_commits(cmd: list[str]) -> None:
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR)
    returncode: int | None = None
    try:
        while returncode is None:
            try:
                returncode = proc.wait(timeout=DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                volume.commit()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        volume.commit()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(extras=["remote"])
)
image = image.run_commands(
    "uv pip install --system --index-url https://download.pytorch.org/whl/cu124 'torch>=2.5,<2.7'"
)

if NETRC_PATH.is_file():
    image = image.add_local_file(NETRC_PATH, remote_path="/root/.netrc", copy=True)

image = image.add_local_dir(".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns())

app = modal.App(APP_NAME)

function_secrets = []
secret_env = {}
if os.environ.get("WANDB_API_KEY"):
    secret_env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
if os.environ.get("OPENAI_API_KEY"):
    secret_env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
if secret_env:
    function_secrets.append(modal.Secret.from_dict(secret_env))

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
    "HF_HOME": f"{VOLUME_PATH}/hf",
    "HF_DATASETS_CACHE": f"{VOLUME_PATH}/hf/datasets",
}
gpu_env = {**env, "REQUIRE_CUDA": "1"}


def _train_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args, default_output_dir="runs/default")
    _assert_wandb_credentials_available_if_needed(normalized_args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.train", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def _reward_model_train_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args, default_output_dir="runs/reward_model_default")
    _assert_wandb_credentials_available_if_needed(normalized_args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.reward_model.train", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def _rm_grpo_train_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args, default_output_dir="runs/rm_grpo_default")
    _assert_wandb_credentials_available_if_needed(normalized_args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.online.train_rm_grpo", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def _eval_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.eval", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def _reward_model_eval_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.reward_model.eval", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def _sample_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.sample", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)

def _build_policy_vs_base_judge_inputs_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.build_policy_vs_base_judge_inputs", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def _build_policy_submission_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.build_policy_submission", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def _judge_candidates_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.judge_candidates", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def _build_reward_model_submission_entrypoint(*args: str) -> None:
    normalized_args = _normalize_args(args)
    cmd = ["python", "-u", "-m", "llm_rl_final_proj.build_reward_model_submission", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="H100",
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def train_remote(*args: str) -> None:
    _train_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="H100",
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def reward_model_train_remote(*args: str) -> None:
    _reward_model_train_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="H100",
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def rm_grpo_train_remote(*args: str) -> None:
    _rm_grpo_train_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="H200",
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def rm_grpo_train_remote_h200(*args: str) -> None:
    _rm_grpo_train_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="A100-40GB",
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def train_remote_a100(*args: str) -> None:
    _train_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=6 * 60 * 60,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=4.0,
    memory=32768,
)
def eval_remote(*args: str) -> None:
    _eval_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=6 * 60 * 60,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=4.0,
    memory=32768,
)
def reward_model_eval_remote(*args: str) -> None:
    _reward_model_eval_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=6 * 60 * 60,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=4.0,
    memory=32768,
)
def sample_remote(*args: str) -> None:
    _sample_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=6 * 60 * 60,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=4.0,
    memory=32768,
)
def build_policy_vs_base_judge_inputs_remote(*args: str) -> None:
    _build_policy_vs_base_judge_inputs_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=6 * 60 * 60,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu=INFERENCE_GPU,
    cpu=4.0,
    memory=32768,
)
def build_policy_vs_base_judge_inputs_remote_a100(*args: str) -> None:
    _build_policy_vs_base_judge_inputs_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=6 * 60 * 60,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="H200",
    cpu=4.0,
    memory=32768,
)
def build_policy_vs_base_judge_inputs_remote_h200(*args: str) -> None:
    _build_policy_vs_base_judge_inputs_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="A100-40GB",
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def build_policy_submission_remote(*args: str) -> None:
    _build_policy_submission_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="A100-40GB",
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def judge_candidates_remote(*args: str) -> None:
    _judge_candidates_entrypoint(*args)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=gpu_env,
    image=image,
    secrets=function_secrets,
    gpu="A100-40GB",
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def build_reward_model_submission_remote(*args: str) -> None:
    _build_reward_model_submission_entrypoint(*args)


@app.local_entrypoint()
def main(*args: str) -> None:
    if _is_wandb_enabled(args) and not NETRC_PATH.is_file() and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "W&B logging is enabled (default), but no credentials were detected locally. "
            "Run `uvx wandb login` (creates ~/.netrc), or export WANDB_API_KEY before modal run, "
            "or pass `--no-wandb_enabled`."
        )
    train_remote.remote(*args)
