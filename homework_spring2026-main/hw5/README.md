# Homework 5: Offline RL

## Setup

For general setup and Modal instructions, see Homework 1's README.

## Examples

Here are some example commands. Run them in the `hw5` directory.

* To run on a local machine:
  ```bash
  uv run src/scripts/run.py --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```


* To run on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```
  * You may request a different GPU type, CPU count, and memory size by changing variables in `src/scripts/modal_run.py`
  * Use `modal run --detach` to keep your job running in the background.
  * Re-running the same command will resume from the latest checkpoint in `exp/` if the job was preempted.
  * Use different seeds or explicitly set `exp_name` if you want to force a separate fresh run with the same arguments.


* To run 4 jobs on a single GPU in parallel on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=30" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000"
  ```

* To download logs and checkpoints from Modal:
  ```bash
  mkdir -p exp
  uv run modal volume get hw5-offline-rl-volume / exp
  ```
