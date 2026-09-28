import time
import argparse
import yaml
import os

from agents.dqn_agent import DQNAgent
from configs import dqn_config

import gym
import numpy as np
import torch
from infrastructure import pytorch_util as ptu
import tqdm

from infrastructure import utils
from infrastructure.log_utils import Logger, setup_wandb, dump_log
from infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"](eval=False)
    eval_env = config["make_env"](eval=True)
    render_env = config["make_env"](eval=True, render=True)
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = env.spec.max_episode_steps

    observation = None

    # Replay buffer
    if len(env.observation_space.shape) == 3:
        stacked_frames = True
        frame_history_len = env.observation_space.shape[0]
        assert frame_history_len == 4, "only support 4 stacked frames"
        replay_buffer = MemoryEfficientReplayBuffer(
            frame_history_len=frame_history_len
        )
    elif len(env.observation_space.shape) == 1:
        stacked_frames = False
        replay_buffer = ReplayBuffer()
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.observation_space.shape}"
        )

    def reset_env_training():
        nonlocal observation

        observation = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.on_reset(observation=observation[-1, ...])

    reset_env_training()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        epsilon = exploration_schedule.value(step)

        # TODO(Section 2.4): Compute action
        action = None
        # ENDTODO

        next_observation, reward, done, info = env.step(action)
        next_observation = np.asarray(next_observation)

        truncated = info.get("TimeLimit.truncated", False)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # We're using the memory-efficient replay buffer,
            # so we only insert next_observation (not observation)
            replay_buffer.insert(
                action=action,
                reward=reward,
                done=done and not truncated,
                next_observation=next_observation[-1, ...],
            )
        else:
            # We're using the regular replay buffer
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                done=done and not truncated,
                next_observation=next_observation,
            )

        # Handle episode termination
        if done:
            reset_env_training()

            logger.log({
                "Train_EpisodeReturn": info["episode"]["r"],
                "Train_EpisodeLen": info["episode"]["l"],
            }, step)
        else:
            observation = next_observation

        # Main DQN training loop
        if step >= config["learning_starts"]:
            # TODO(Section 2.4): Sample config["batch_size"] samples from the replay buffer
            batch = None
            # ENDTODO

            batch = ptu.from_numpy(batch)

            # TODO(Section 2.4): Train the agent.
            update_info = None
            # ENDTODO

            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                if step % args.eval_interval != 0:
                    logger.log(update_info, step)

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
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
            if step >= config["learning_starts"]:
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
    return dqn_config.configs[base_config_name](**config_kwargs)


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

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
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
