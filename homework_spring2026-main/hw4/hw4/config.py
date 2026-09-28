from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_dir: str = "runs/default"
    task: str = "format_copy"  # format_copy | math_hard

    seed: int = 0
    steps: int = 1200

    batch_size: int = 8
    group_size: int = 6

    # Sampling
    min_new_tokens: int = 1
    max_new_tokens: int = 512
    max_prompt_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    # LoRA optimization (base model stays frozen)
    lr: float = 3e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 100
    grad_accum_steps: int = 1

    # RL update hyperparameters
    algo: str = "grpo"      # reinforce | grpo
    ppo_epochs: int = 1  # used by GRPO; REINFORCE requires 1
    minibatch_size: int = 8
    clip_eps: float = 0.1
    kl_coef: float = 0.05
    max_grad_norm: float = 0.5
    adv_clip: float = 5.0
    normalize_advantages: bool = False

    # LoRA adapter config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"

    # Memory/speed
    grad_checkpointing: bool = True
    rollout_on_cpu: bool = True # rollouts computed on gpu, but offloaded to cpu for memory saving (until needed)
    cuda_empty_cache_interval: int = 0

    # Logging / checkpointing
    wandb_project: str = "llm-rl-hw4"
    wandb_name: str = "run"
    wandb_enabled: bool = True
    sample_log_interval: int = 10
    sample_markdown_log_interval: int = 1
    sample_log_n: int = 3
    sample_log_max_chars: int = 3000

    eval_interval: int = 100
    save_interval: int = 100

    # Eval limits
    format_copy_eval_n: int = 64
    math_hard_eval_n: int = 512
    eval_batch_size: int = 32
