import sys
from pathlib import Path

import modal

from scripts.run_dqn import main


APP_NAME = "hw3-ql"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/vol"
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
DEFAULT_MEMORY = 4096  # MB
volume = modal.Volume.from_name("hw3-ql-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""

    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    # Always exclude exp/ directory (logs should only go to the Modal volume)
    patterns: list[str] = ["**/exp/**"]
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# Build a container image with the project's dependencies using uv.
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0", "swig").uv_sync()
# Copy .netrc for wandb logging.
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
# Copy the current directory.
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
}


@app.function(volumes={VOLUME_PATH: volume}, timeout=60 * 60 * 5, env=env, image=image, gpu=DEFAULT_GPU, cpu=DEFAULT_CPU, memory=DEFAULT_MEMORY)
def hw3_dqn_remote(*args: str) -> None:
    import os

    os.chdir(PROJECT_DIR)
    exp_vol = Path(VOLUME_PATH) / "exp"
    exp_vol.mkdir(parents=True, exist_ok=True)

    exp_link = Path(PROJECT_DIR) / "exp"
    if exp_link.is_dir() and not exp_link.is_symlink():
        import shutil
        shutil.rmtree(exp_link)
    elif exp_link.exists() or exp_link.is_symlink():
        exp_link.unlink()
    exp_link.symlink_to(exp_vol)

    sys.argv = ["run_dqn.py"] + list(args)
    main()
    volume.commit()
