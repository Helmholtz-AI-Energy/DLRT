from __future__ import annotations

import mlflow.pytorch
import pytorch_warmup as warmup
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim.lr_scheduler as lr_schedules
import torch.utils.data.distributed
import torchvision.models as models
from mpi4py import MPI
from PIL import ImageFile
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

import dlrt
from networks import comm
from networks import datasets as dsets

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()

ImageFile.LOAD_TRUNCATED_IMAGES = True
console = Console(width=140)


def get_lr_schedules(config, optim, len_ds=None):
    """
    Get learning rate schedules from config files

    Parameters
    ----------
    config
    optim

    Returns
    -------

    """
    sched_name = getattr(lr_schedules, config["lr_schedule"]["name"])
    sched_params = config["lr_schedule"]["params"]
    if config["lr_schedule"]["name"] == "ExponentialLR":
        sched_params["last_epoch"] = config["epochs"] - config["start_epoch"]
    elif config["lr_schedule"]["name"] == "CosineAnnealingLR":
        # sched_params["last_epoch"] = config['epochs'] - config['start_epoch']
        sched_params["T_max"] = len_ds
    elif config["lr_schedule"]["name"] == "CosineAnnealingWarmRestarts":
        sched_params["T_0"] = len_ds
    elif config["lr_schedule"]["name"] == "CyclicLR":
        sched_params["max_lr"] = config["learning_rate"]
        sched_params["step_size_up"] = len_ds

    scheduler = sched_name(optim, **sched_params)
    wup_sched_name = getattr(warmup, config["lr_warmup"]["name"])
    wup_sched_params = config["lr_warmup"]["params"]
    warmup_scheduler = wup_sched_name(optim, **wup_sched_params)
    return scheduler, warmup_scheduler
