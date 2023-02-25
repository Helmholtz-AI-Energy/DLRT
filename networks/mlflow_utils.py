from __future__ import annotations

import os
import subprocess
import time

import mlflow
import torch.distributed as dist

__all__ = ["setup_mlflow"]

from mlflow.entities import Experiment


def print0(*args, sep=" ", end="\n", file=None):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(*args, sep=sep, end=end, file=file)
    elif dist.is_initialized():
        return
    else:
        print(*args, sep=sep, end=end, file=file)


def setup_mlflow(config: dict, verbose: bool = False, rank: int = None) -> Experiment | None:
    """Setup MLFlow server, connect to it, and set the experiment

    Parameters
    ----------
    config: dict
        Config dictionary
    verbose: bool
        if this should print the mlflow server on rank0
    rank: int, optional
        process rank

    exp_id: int
        MLFlow experiment ID
    """
    if rank is not None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    if rank != 0:
        return
    restart_mlflow_server(config)
    # Connect to the MLFlow client for tracking this training - only on rank0!
    mlflow_server = f"http://127.0.0.1:{config['mlflow']['port']}"
    mlflow.set_tracking_uri(mlflow_server)
    if verbose:
        print0(f"MLFlow connected to server {mlflow_server}")

    experiment = mlflow.set_experiment(f"{config['dataset']}-{config['arch']}")
    return experiment


def restart_mlflow_server(config: dict):
    """Kill any existing mlflow servers and restart them

    Parameters
    ----------
    config: dict
        Config dictionary
    """

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # kill mlflow and start the server
    processname = "gunicorn"
    tmp = os.popen("ps -Af").read()
    proccount = tmp.count(processname)
    if proccount == 0:
        subprocess.Popen(["pkill", "-f", "gunicorn"])
        print0("Starting MLFlow server")
        mlflow_server_cmd = [
            "/usr/local/bin/mlflow",
            "server",
            "--backend-store-uri",
            f"{config['mlflow']['tracking_uri']}",
            "--default-artifact-root",
            f"{config['mlflow']['artifact_location']}",
            "--port",
            f"{config['mlflow']['port']}",
        ]
        print("mlflow cmd", mlflow_server_cmd)
        _ = subprocess.Popen(mlflow_server_cmd)
        time.sleep(2)
