import argparse
import json
import os
import tempfile
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Optimizer
import tqdm
import wandb

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log


def get_run_name(args: argparse.Namespace) -> str:
    if args.exp_name is not None:
        return args.exp_name

    exp_name = f"sd{args.seed}_{args.base_config}_{args.env_name}"
    if args.alpha is not None:
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.expectile is not None:
        exp_name = f"{exp_name}_e{args.expectile}"
    return exp_name


def load_checkpoint(save_dir: str, agent=None) -> tuple[int, bool]:
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        return 0, False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("status") == "completed":
        return 0, True

    if agent is None:
        return checkpoint["next_step"], False

    agent.load_state_dict(checkpoint["agent_state_dict"])
    for name, state_dict in checkpoint["optimizer_states"].items():
        optimizer = getattr(agent, name, None)
        if isinstance(optimizer, Optimizer):
            optimizer.load_state_dict(state_dict)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(ptu.device)

    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
    return checkpoint["next_step"], False


def save_checkpoint(
    save_dir: str,
    agent,
    next_step: int,
    status: str,
    checkpoint_callback: Optional[Callable[[], None]] = None,
) -> None:
    optimizer_states = {}
    for name, value in vars(agent).items():
        if isinstance(value, Optimizer):
            optimizer_states[name] = value.state_dict()

    checkpoint = {
        "next_step": next_step,
        "status": status,
        "agent_state_dict": agent.state_dict(),
        "optimizer_states": optimizer_states,
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()

    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    with tempfile.NamedTemporaryFile(dir=save_dir, suffix=".tmp", delete=False) as f:
        temp_path = f.name
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    if checkpoint_callback is not None:
        checkpoint_callback()


def run_training_loop(
    config: dict,
    train_logger,
    eval_logger,
    args: argparse.Namespace,
    checkpoint_callback: Optional[Callable[[], None]] = None,
):
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # Make the gymnasium environment
    env, dataset = config["make_env_and_dataset"]()

    example_batch = dataset.sample(1)
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        example_batch['observations'].shape[1:],
        example_batch['actions'].shape[-1],
        **config["agent_kwargs"],
    )

    start_step, is_completed = load_checkpoint(args.save_dir, agent)
    if is_completed:
        return

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    step_iterator = range(start_step, config["training_steps"] + 1)
    if start_step > 0:
        remaining_steps = config["training_steps"] - start_step + 1
        print(f"Resuming from step {start_step} with {remaining_steps} steps remaining.")
    progress_bar = tqdm.tqdm(
        step_iterator,
        total=config["training_steps"] + 1,
        initial=start_step,
        dynamic_ncols=True,
    )

    for step in progress_bar:
        # Train with offline RL
        batch = dataset.sample(config["batch_size"])

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            train_logger.log(metrics, step=step)

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            successes = [t["episode_statistics"]["s"] for t in trajectories]

            eval_logger.log(
                {
                    "eval/success_rate": float(np.mean(successes)),
                },
                step=step,
            )
            save_checkpoint(
                args.save_dir,
                agent,
                next_step=step + 1,
                status="running",
                checkpoint_callback=checkpoint_callback,
            )

    dump_log(agent, train_logger, eval_logger, config, args.save_dir)
    save_checkpoint(
        args.save_dir,
        agent,
        next_step=config["training_steps"] + 1,
        status="completed",
        checkpoint_callback=checkpoint_callback,
    )


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--training_steps", type=int, default=1000000)  # Should be less than or equal to 1M to pass autograder
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder

    parser.add_argument("--expectile", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args, checkpoint_callback: Optional[Callable[[], None]] = None):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder
    exp_name = get_run_name(args)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    _, is_completed = load_checkpoint(args.save_dir)
    if is_completed:
        print(f"Run already completed at {args.save_dir}; skipping.")
        return

    config = configs.configs[args.base_config](args.env_name)

    # Set common config values from args for autograder
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['training_steps'] = args.training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['exp_name'] = exp_name

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha

    wandb_run_path = os.path.join(args.save_dir, "wandb_run.json")
    wandb_run_id = None
    if os.path.exists(wandb_run_path):
        with open(wandb_run_path) as f:
            wandb_run_id = json.load(f).get("run_id")
    wandb_resume = "must" if wandb_run_id is not None else None
    wandb_run = setup_wandb(
        project='cs285_hw5',
        name=exp_name,
        group=args.run_group,
        run_id=wandb_run_id,
        resume=wandb_resume,
        config=config,
    )
    temp_fd, temp_path = tempfile.mkstemp(dir=args.save_dir, suffix=".wandb.tmp")
    os.close(temp_fd)
    try:
        with open(temp_path, "w") as f:
            json.dump({"run_id": wandb_run.id}, f)
        os.replace(temp_path, wandb_run_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    try:
        run_training_loop(
            config,
            train_logger,
            eval_logger,
            args,
            checkpoint_callback=checkpoint_callback,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
