import multiprocessing as mp
import shlex

import modal

from scripts.run import setup_arguments, main


def _worker(job_str: str):
    job_args_list = shlex.split(job_str)
    assert job_args_list[0] == 'JOB'
    del job_args_list[0]  # Delete the dummy "JOB" prefix
    print(job_args_list)
    job_args = setup_arguments(args=job_args_list)
    volume = modal.Volume.from_name("hw5-offline-rl-volume", create_if_missing=True)

    main(job_args, checkpoint_callback=volume.commit)


def main_njobs(job_specs, njobs: int, start_method: str = "spawn"):
    try:
        mp.set_start_method(start_method, force=True)
    except RuntimeError:
        pass

    with mp.Pool(processes=njobs) as pool:
        pool.starmap(_worker, [(spec,) for spec in job_specs])
