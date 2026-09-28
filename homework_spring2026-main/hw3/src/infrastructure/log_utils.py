import copy
import json
import os
import pickle
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import torch
from torch import nn
import wandb
from PIL import Image, ImageEnhance


class Logger:
    """Logger for logging metrics."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)
        self.rows = []

    def log(self, row, step):
        row['step'] = step
        filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}

        # Check for new columns
        new_keys = [k for k in filtered_row if self.header is None or k not in self.header]
        if self.header is None:
            self.header = list(filtered_row.keys())
            self.file = open(self.path, 'w')
            self.file.write(','.join(self.header) + '\n')
        elif new_keys:
            # Expand header with new columns and rewrite CSV
            self.header.extend(new_keys)
            self.file.close()
            self.file = open(self.path, 'w')
            self.file.write(','.join(self.header) + '\n')
            for prev_row in self.rows:
                prev_filtered = {k: v for k, v in prev_row.items() if not isinstance(v, self.disallowed_types)}
                self.file.write(','.join([str(prev_filtered.get(k, '')) for k in self.header]) + '\n')

        self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

        wandb.log(row, step=step)
        self.rows.append(copy.deepcopy(row))

    def log_scalar(self, scalar, name, step):
        """Log a single scalar value."""
        wandb.log({name: scalar}, step=step)
        # Don't add to rows for individual scalars - they will be batched in the log() method

    def log_trajs_as_videos(self, trajs, step, max_videos_to_save=2, fps=10, video_title='video'):
        videos = [traj['image_obs'] for traj in trajs][:max_videos_to_save]
        video = get_wandb_video(videos, fps=fps)
        wandb.log({video_title: video}, step=step)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):
        """Alias for log_trajs_as_videos for compatibility."""
        self.log_trajs_as_videos(paths, step, max_videos_to_save, fps, video_title)

    def flush(self):
        """Flush the log file."""
        if self.file is not None:
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def remove_functions(obj):
    if isinstance(obj, dict):
        return {
            k: remove_functions(v)
            for k, v in obj.items()
            if not callable(v)
        }
    elif isinstance(obj, list):
        return [remove_functions(v) for v in obj if not callable(v)]
    elif callable(obj):
        return None
    else:
        return obj


def dump_log(agent: nn.Module, logger: Logger, args, save_dir: str):
    """Dump the log to a pkl file."""
    cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    config = vars(args)
    config = remove_functions(config)

    filtered_rows = [r for r in logger.rows if 'Train_EpisodeReturn' not in r]

    data = {
        'log': filtered_rows,
        'log_hash': hash(json.dumps(str(logger.rows), sort_keys=True)),
        'config': config,
        'config_hash': hash(json.dumps(str(config), sort_keys=True)),
        'time': cur_time,
    }

    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(config, f)
    with open(os.path.join(save_dir, f'log.pkl'), 'wb') as f:
        pickle.dump(data, f)

    torch.save(agent.state_dict(), os.path.join(save_dir, 'agent.pt'))


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
    config=None,
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    # Truncate tag to max 64 characters (WandB limit)
    if group is not None and len(group) > 64:
        tags = [group[:64]]
    else:
        tags = [group] if group is not None else None

    init_kwargs = dict(
        config=config,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')
