import copy
import csv
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
        self._load_existing_rows()

    @staticmethod
    def _parse_scalar(value):
        if value == "":
            return value

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            return value

    def _load_existing_rows(self):
        if not os.path.exists(self.path):
            return

        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            self.header = reader.fieldnames
            if self.header is None:
                return

            self.rows = [
                {k: self._parse_scalar(v) for k, v in row.items()}
                for row in reader
            ]

    def log(self, row, step):
        row['step'] = step
        filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        if self.file is None:
            mode = 'a' if self.header is not None else 'w'
            self.file = open(self.path, mode)
            if self.header is None:
                self.header = list(filtered_row.keys())
                self.file.write(','.join(self.header) + '\n')
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

        wandb.log(row, step=step)
        self.rows.append(copy.deepcopy(row))

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


def dump_log(agent: nn.Module, train_logger: Logger, eval_logger: Logger, config: dict, save_dir: str):
    """Dump the log to a pkl file."""
    cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    config = remove_functions(config)

    data = {
        'train': train_logger.rows,
        'train_hash': hash(json.dumps(train_logger.rows, sort_keys=True)),
        'eval': eval_logger.rows,
        'eval_hash': hash(json.dumps(eval_logger.rows, sort_keys=True)),
        'config': config,
        'config_hash': hash(json.dumps(config, sort_keys=True)),
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
    run_id=None,
    resume=None,
    mode='online',
    config=None,
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=config,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        id=run_id,
        resume=resume,
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
