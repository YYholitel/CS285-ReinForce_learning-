from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    eval_split: str = "test_prefs"
    generation_split: str = "test_gen"
    output_dir: str = "runs/default"

    seed: int = 0
    algo: str = "dpo"  # dpo | ipo | aot
    num_train_epochs: float = 1.0
    max_steps: int = 0  # 0 means no explicit cap; iterate over epochs.

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    beta: float = 0.1

    max_prompt_tokens: int = 700
    max_response_tokens: int = 512

    train_limit: int = 0
    eval_limit: int = 512
    generation_eval_limit: int = 64
    generation_eval_max_new_tokens: int = 256
    generation_eval_temperature: float = 0.0
    generation_eval_top_p: float = 1.0
    generation_eval_every: int = 200

    eval_interval: int = 200
    save_interval: int = 200

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    grad_checkpointing: bool = True

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "run"
    wandb_enabled: bool = True

    sample_log_n: int = 8
    sample_log_max_chars: int = 3000
