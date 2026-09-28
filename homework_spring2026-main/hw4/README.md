# HW 4: LLM RL

This repository is set up to run on Modal by default via `scripts/modal_train.py`. The default Modal GPU is an `H100`.

## Quickstart

```bash
uv sync
uv run modal token new
uvx wandb login
```

All training commands below are intended to be run from the repository root.

## Required Runs

### Format Copy + GRPO

```bash
uv run modal run --detach scripts/modal_train.py -- \
  --task format_copy \
  --algo grpo \
  --output_dir /vol/runs/modal_format_copy_grpo \
  --steps 51 \
  --batch_size 8 \
  --group_size 6 \
  --min_new_tokens 1 \
  --max_new_tokens 24 \
  --lr 3e-5 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --grad_accum_steps 6 \
  --clip_eps 0.2 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name format_copy_grpo \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 6 \
  --eval_interval 50 \
  --save_interval 50 \
  --warmup_steps 10
```

### Format Copy + REINFORCE

```bash
uv run modal run --detach scripts/modal_train.py -- \
  --task format_copy \
  --algo reinforce \
  --output_dir /vol/runs/modal_format_copy_reinforce \
  --steps 51 \
  --batch_size 8 \
  --group_size 6 \
  --min_new_tokens 1 \
  --max_new_tokens 24 \
  --lr 3e-5 \
  --minibatch_size 8 \
  --grad_accum_steps 6 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name format_copy_reinforce \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 6 \
  --eval_interval 50 \
  --save_interval 50 \
  --warmup_steps 10
```

### Math Hard + REINFORCE

```bash
uv run modal run --detach scripts/modal_train.py -- \
  --task math_hard \
  --algo reinforce \
  --output_dir /vol/runs/modal_math_hard_reinforce \
  --steps 201 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --minibatch_size 8 \
  --grad_accum_steps 8 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name math_hard_reinforce \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100
```

### Math Hard + GRPO

```bash
uv run modal run --detach scripts/modal_train.py -- \
  --task math_hard \
  --algo grpo \
  --output_dir /vol/runs/modal_math_hard_grpo \
  --steps 501 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --grad_accum_steps 8 \
  --clip_eps 0.2 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name math_hard_grpo \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100
```

## Build the Gradescope Bundle

Pass `--run_dir` once for each completed run you want to include. You can build and submit a partial bundle before all four required runs are finished; missing runs will simply show up as missing / zero-credit on Gradescope until you add them later.

```bash
uv run modal run scripts/modal_train.py::bundle_submission_remote -- \
  --run_dir /vol/runs/modal_format_copy_grpo \
  --run_dir /vol/runs/modal_format_copy_reinforce \
  --run_dir /vol/runs/modal_math_hard_grpo \
  --run_dir /vol/runs/modal_math_hard_reinforce \
  --output_dir /vol/submissions/hw4_gradescope_submission \
  --overwrite
```

```bash
uv run modal volume get hw4-llm-rl-volume /submissions/hw4_gradescope_submission.zip .
```

## Package for Gradescope

1. Unzip `hw4_gradescope_submission.zip`.
2. Put your homework repo folder `hw4/` and the unzipped `hw4_gradescope_submission/` folder side by side in the same parent directory.
3. Zip those two folders together directly, with no outer wrapper directory.

The uploaded zip should look like this at the top level:

```text
hw4_submission.zip
  hw4/
    ...
  hw4_gradescope_submission/
    ...
```

If your current working directory is the parent directory containing both folders, one convenient command is:

```bash
zip -r hw4_submission.zip hw4 hw4_gradescope_submission \
  -x "hw4/.venv/*" "hw4/.git/*" "hw4/**/__pycache__/*" "*.pyc" ".DS_Store"
```
