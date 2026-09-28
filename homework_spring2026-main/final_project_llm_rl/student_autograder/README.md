# Student Local Autograder

This directory contains the local GPT-5.4 evaluation script used for development.

## Usage

From the repository root, run:

```bash
export OPENAI_API_KEY=...
uv run python student_autograder/run_local_autograder.py \
  --submission_dir llm_rl_final_proj_public_submission \
  --output_json student_autograder_results.json
```

The local autograder uses the same evaluation files and thresholds as Gradescope.

## What it grades
- Part 1 reward model on `reward_model/public_test_pref_scores.jsonl`
- Part 1 policy methods on the repository `128`-prompt evaluation file
- Part 2 offline best on the same `128`-prompt evaluation file
- Part 2 online best on the same `128`-prompt evaluation file

## Requirements
Install the Python dependencies with:

```bash
uv pip install -r student_autograder/requirements.txt
```
