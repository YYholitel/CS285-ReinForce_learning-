import os
import time
import yaml
import argparse

from agents.sac_agent import SoftActorCritic
from configs import sac_config
from infrastructure.replay_buffer import ReplayBuffer

import gym
import numpy as np
import torch
from infrastructure import pytorch_util as ptu
import tqdm

from infrastructure import utils
from infrastructure.log_utils import Logger, setup_wandb, dump_log


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"](eval=True)
    render_env = config["make_env"](eval=True, render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "SAC only supports continuous action spaces."

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 10

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # TODO(Section 3.1): Select an action
            action = None
            # ENDTODO

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, info = env.step(action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            logger.log({
                "Train_EpisodeReturn": info["episode"]["r"],
                "Train_EpisodeLen": info["episode"]["l"],
            }, step)
            observation = env.reset()
        else:
            observation = next_observation

        # Train the agent
        if step >= config["training_starts"]:
            # TODO(Section 3.1): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = None
            update_info = None
            # ENDTODO

            # Logging
            if step % args.log_interval == 0:
                if step % args.eval_interval != 0:
                    logger.log(update_info, step)

        # Run evaluation
        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            eval_metrics = {
                "Eval_AverageReturn": np.mean(returns),
                "Eval_StdReturn": np.std(returns),
                "Eval_MaxReturn": np.max(returns),
                "Eval_MinReturn": np.min(returns),
                "Eval_AverageEpLen": np.mean(ep_lens),
            }

            # Merge training metrics if available
            if step >= config["training_starts"]:
                eval_metrics.update(update_info)
            logger.log(eval_metrics, step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )

            # Save checkpoint periodically
            dump_log(agent, logger, args, os.path.dirname(logger.path))

    dump_log(agent, logger, args, os.path.dirname(logger.path))


def make_config(config_file: str) -> dict:
    with open(config_file, "r") as f:
        config_kwargs = yaml.safe_load(f)

    base_config_name = config_kwargs.pop("base_config")
    return sac_config.configs[base_config_name](**config_kwargs)


def make_logger(config: dict, args: argparse.Namespace) -> Logger:
    logdir = "{}_sd{}_{}".format(
        config["log_name"], args.seed, time.strftime("%Y%m%d_%H%M%S")
    )
    logdir = os.path.join("exp", logdir)
    os.makedirs(logdir, exist_ok=True)

    # Setup WandB
    wandb_config = {**config, **vars(args)}
    setup_wandb(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=config["log_name"],
        name=logdir.split("/")[-1],
        mode="online",
        config=wandb_config,
    )

    return Logger(os.path.join(logdir, "log.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    # WandB arguments
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="hw3")

    args = parser.parse_args()

    config = make_config(args.config_file)
    logger = make_logger(config, args)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
