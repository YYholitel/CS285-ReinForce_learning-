# Offline to Online RL Final Project

## Setup

For general setup and Modal instructions, see Homework 1's README.

## Examples

Here are some example commands. Run them in the `final_project_offline_online` directory.

* To run on a local machine:
  ```bash
  uv run src/scripts/run.py --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```


* To run on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```
  * You may request a different GPU type, CPU count, and memory size by changing variables in `src/scripts/modal_run.py`
  * Use `modal run --detach` to keep your job running in the background.


* To run 4 jobs on a single GPU in parallel on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=30" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000"
  ```
