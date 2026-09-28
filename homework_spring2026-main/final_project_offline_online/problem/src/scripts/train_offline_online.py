import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log
from infrastructure.replay_buffer import ReplayBuffer


def run_offline_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, start_step: int = 0):
    """
    Run offline training loop
    """
    # TODO(student): Implement offline training loop
    
    return ...

def run_online_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, agent_path: str, start_step: int = 0):
    """
    Run online training loop
    """
    # TODO(student): Implement online training loop
    return ...


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--offline_training_steps", type=int, default=500000)  # Should be 500k to pass the autograder
    parser.add_argument("--online_training_steps", type=int, default=100000)  # Should be 100k to pass the autograder
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder
    
    # Online retention of offline data
    # TODO(student): If desired, add arguments for online retention of offline data
    
    # WSRL
    # TODO (student): If desired, add arguments for WSRL
    

    # IFQL
    parser.add_argument("--expectile", type=float, default=None)

    # FQL / QSM
    parser.add_argument("--alpha", type=float, default=None)

    # QSM
    parser.add_argument("--inv_temp", type=float, default=None)

    # DSRL
    parser.add_argument("--noise_scale", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    config = configs.configs[args.base_config](args.env_name)

    # Set common config values from args for autograder
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['offline_training_steps'] = args.offline_training_steps
    config['online_training_steps'] = args.online_training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['replay_buffer_capacity'] = args.replay_buffer_capacity
    
    # TODO(student): If necessary, add additional config values

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
    if args.online_training_steps > 0:
        exp_name = f"{exp_name}_online"
    if args.offline_training_steps > 0:
        exp_name = f"{exp_name}_offline"

    setup_wandb(project='cs185_default_project', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    start_step = 0
    if args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        # TODO(student): Implement offline training loop
        # Hint: You might consider passing the agent's path to the online training loop
        ... = run_offline_training_loop(config, train_logger, eval_logger, args, start_step=0)
        start_step = args.offline_training_steps
        
    
    if args.online_training_steps > 0:
        print(f"Running online training loop with {args.online_training_steps} steps")
        # TODO(student): Implement online training loop
        run_online_training_loop(config, train_logger, eval_logger, args, ..., start_step=start_step)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
