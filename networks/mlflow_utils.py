import os
import subprocess
import time

import mlflow
from mlflow.tracking import MlflowClient
import torch
import torch.distributed as dist

__all__ = ["setup_mlflow"]


def print0(*args, sep=' ', end='\n', file=None):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(*args, sep=sep, end=end, file=file)
    elif dist.is_initialized():
        return
    else:
        print(*args, sep=sep, end=end, file=file)


def setup_mlflow(config: dict) -> int:
    """Setup MLFlow server, connect to it, and set the experiment

    Parameters
    ----------
    config: dict
        Config dictionary

    exp_id: int
        MLFlow experiment ID
    """

    restart_mlflow_server(config)

    # Connect to the MLFlow client for tracking this training
    mlflow_server = f"http://127.0.0.1:{config['mlflow']['port']}"
    mlflow.set_tracking_uri(mlflow_server)
    print0(f"MLFlow connected to server {mlflow_server}")

    # Setup experiment
    exp_id = set_mlflow_experiment(config)
    return exp_id


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
