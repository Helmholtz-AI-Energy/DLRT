from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from enum import Enum
from pathlib import Path

import comm
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
import yaml
from mpi4py import MPI
from PIL import ImageFile
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from .. import datasets as dsets
from .. import mlflow_utils as mlfutils
from .. import optimizer as opt

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()

ImageFile.LOAD_TRUNCATED_IMAGES = True
console = Console(width=140)


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="config file to load parameters from",
)

best_acc1 = 0


def main(config):  # noqa: C901
    if "seed" in config:
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])

    # initialize the torch process group across all processes
    print("comm init")
    try:
        if int(os.environ["SLURM_NTASKS"]) > 1 or int(os.environ["OMPI_COMM_WORLD_SIZE"]) > 1:
            comm.init(method="nccl-slurm")
            config["world_size"] = dist.get_world_size()
            config["rank"] = dist.get_rank()
        else:
            config["world_size"] = 1
            config["rank"] = 0
    except KeyError:
        try:
            if int(os.environ["OMPI_COMM_WORLD_SIZE"]) > 1:
                comm.init(method="nccl-slurm")
                config["world_size"] = dist.get_world_size()
                config["rank"] = dist.get_rank()
            else:
                config["world_size"] = 1
                config["rank"] = 0
        except KeyError:
            config["world_size"] = 1
            config["rank"] = 0

    if config["rank"] == 0:
        mlflow.log_params({"world_size": config["world_size"], "rank": config["rank"]})

    # create model
    if config["pretrained"]:
        print(f"=> using pre-trained model '{config['arch']}'")
        model = models.__dict__[config["arch"]](pretrained=True)
    else:
        print(f"=> creating model '{config['arch']}'")
        model = models.__dict__[config["arch"]]()

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if dist.is_initialized():
        config["gpu"] = dist.get_rank() % torch.cuda.device_count()  # only 4 gpus/node
        print(config["gpu"])
    else:
        config["gpu"] = 0
    torch.cuda.set_device(config["gpu"])
    model.cuda(config["gpu"])
    device = torch.device(f"cuda:{config['gpu']}")

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        config["learning_rate"],
        momentum=config["optimizer"]["params"]["momentum"],
        weight_decay=config["optimizer"]["params"]["weight_decay"],
        nesterov=config["optimizer"]["params"]["nesterov"],
    )

    scheduler, warmup_scheduler = opt.get_lr_schedules(config=config, optim=optimizer)

    # log parameters from config
    if config["rank"] == 0:
        mlflow.log_params(config)
        for cat in ["dlrt", "lr_schedule", "lr_warmup", "optimizer"]:
            for k in config[cat]:
                if isinstance(config[cat][k], dict):
                    for k2 in config[cat][k]:
                        mlflow.log_param(f"{cat}-{k}-{k2}", config[cat][k][k2])
                else:
                    mlflow.log_param(f"{cat}-{k}", config[cat][k])

    # optionally resume from a checkpoint
    # TODO: add DLRT checkpointing
    if config["resume"]:
        if os.path.isfile(config["resume"]):
            print(f"=> loading checkpoint: {config['resume']}")
            if config["gpu"] is None:
                checkpoint = torch.load(config["resume"])
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{config['gpu']}"
                checkpoint = torch.load(config["resume"], map_location=loc)
            config["start_epoch"] = checkpoint["epoch"]
            # best_acc1 = checkpoint["best_acc1"]
            # if config["gpu"] is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(config["gpu"])
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    config["resume"],
                    checkpoint["epoch"],
                ),
            )
        else:
            print(f"=> no checkpoint found at: {config['resume']}")

    # Data loading code
    if config["dataset"] == "imagenet":
        dset_dict = dsets.get_imagenet_datasets(
            config["data_location"],
            config["local_batch_size"],
            config["workers"],
        )
    elif config["dataset"] == "cifar10":
        dset_dict = dsets.get_cifar10_datasets(
            config["data_location"],
            config["local_batch_size"],
            config["workers"],
        )
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} not implemented")
    train_loader, train_sampler = dset_dict["train"]["loader"], dset_dict["train"]["sampler"]
    val_loader = dset_dict["val"]["loader"]

    # if config['evaluate']:
    #     validate(val_loader, dlrt_trainer, config)
    #     return
    for epoch in range(config["start_epoch"], config["epochs"]):
        if config["rank"] == 0:
            console.rule(f"Begin epoch {epoch} LR: {optimizer.param_groups[0]['lr']}")
            mlflow.log_metrics(
                metrics={"lr": optimizer.param_groups[0]["lr"]},
                step=epoch,
            )
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

        train_loss = train(
            train_loader,
            optimizer,
            model,
            criterion,
            epoch,
            device,
            config,
            warmup_scheduler=warmup_scheduler,
        )

        # save_selected_weights(model, epoch)
        if rank == 0:
            print(f"Average Training loss across process space: {train_loss}")
        # evaluate on validation set
        _, val_loss = validate(val_loader, model, criterion, config, epoch)
        if rank == 0:
            print(
                f"Average val loss across process space: {val_loss} " f"-> diff: {train_loss - val_loss}",
            )

        with warmup_scheduler.dampening():
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:  # StepLR / ExponentialLR / others
                scheduler.step()


def train(train_loader, optimizer, model, criterion, epoch, device, config, warmup_scheduler):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(train_loader) - 1 and warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                pass

        if (i % config["print_freq"] == 0 or i == len(train_loader) - 1) and config["rank"] == 0:
            # console.rule(f"train step {i}")
            argmax = torch.argmax(output, dim=1).to(torch.float32)
            console.print(
                f"Argmax outputs s "
                f"mean: {argmax.mean().item():.5f}, max: {argmax.max().item():.5f}, "
                f"min: {argmax.min().item():.5f}, std: {argmax.std().item():.5f}",
            )
            progress.display(i + 1)

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if config["rank"] == 0:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        mlflow.log_metrics(
            metrics={
                "train loss": ls,
                "train top1": t1,
                "train top5": t5,
            },
            step=epoch,
        )
    return losses.avg


@torch.no_grad()
def save_selected_weights(network, epoch):
    save_list = [
        "module.conv1.weight",
        "module.fc.weight",
        "module.layer1.1.conv2.weight",
        "module.layer3.1.conv2.weight",
        "module.layer4.0.downsample.0.weight",
    ]
    save_location = Path(
        "/hkfs/work/workspace/scratch/qv2382-dlrt/saved_models/4gpu-svd-tests/normal/resnet18",
    )
    rank = dist.get_rank()

    if rank != 0:
        dist.barrier()
        return
    # save location: resnet18/name/epoch/[u, s, vh, weights
    for n, p in network.named_parameters():
        if n in save_list:
            print(n)
            n_save_loc = save_location / n / str(epoch)
            n_save_loc.mkdir(exist_ok=True, parents=True)
            # todo: full matrices?? -> can always slice later
            if p.data.ndim > 2:
                tosave = p.view(p.shape[0], -1)
            else:
                tosave = p.data
            u, s, vh = torch.linalg.svd(tosave, full_matrices=False)
            torch.save(u, n_save_loc / "u-reduced.pt")
            torch.save(s, n_save_loc / "s-reduced.pt")
            torch.save(vh, n_save_loc / "vh-reduced.pt")
            u, s, vh = torch.linalg.svd(tosave, full_matrices=True)
            torch.save(u, n_save_loc / "u.pt")
            torch.save(s, n_save_loc / "s.pt")
            torch.save(vh, n_save_loc / "vh.pt")
            torch.save(tosave.data, n_save_loc / "p.pt")
    print("finished saving")
    dist.barrier()


def validate(val_loader, model, criterion, config, epoch):
    console.rule("validation")

    def run_validate(loader, base_progress=0):
        rank = 0 if not dist.is_initialized() else dist.get_rank()
        with torch.no_grad():
            end = time.time()
            num_elem = len(loader) - 1
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.cuda(config["gpu"], non_blocking=True)
                target = target.cuda(config["gpu"], non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)
                # argmax = torch.argmax(output.output, dim=1).to(torch.float32)
                # print(
                #     f"output mean: {argmax.mean().item()}, max: {argmax.max().item()}, min: {argmax.min().item()}, std: {argmax.std().item()}",
                # )

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i % config["print_freq"] == 0 or i == num_elem) and rank == 0:
                    argmax = torch.argmax(output, dim=1).to(torch.float32)
                    print(
                        f"output mean: {argmax.mean().item()}, max: {argmax.max().item()}, min: {argmax.min().item()}, std: {argmax.std().item()}",
                    )
                    progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4f", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset)),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * config["world_size"], len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=config["local_batch_size"],
            shuffle=False,
            num_workers=config["workers"],
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    if dist.is_initialized():
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if config["rank"] == 0:
        ls = losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg
        t1 = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        t5 = top5.avg.item() if isinstance(top5.avg, torch.Tensor) else top5.avg
        mlflow.log_metrics(
            metrics={
                "train loss": ls,
                "train top1": t1,
                "train top5": t5,
            },
            step=epoch,
        )

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        # self.avg = self.sum / self.count
        self.avg = total[0] / total[1]  # self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # if self.rank == 0:
        #     # print("\t".join(entries))
        console.print(" ".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if self.rank == 0:
            # print(" ".join(entries))
            console.print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        experiment = mlfutils.setup_mlflow(config, verbose=False)
        # run_id -> adaptive needs to be unique, roll random int?
        # run_name = f"" f"full-rank-everybatch-{os.environ['SLURM_JOBID']}"
        with mlflow.start_run() as run:
            mlflow.log_param("Slurm jobid", os.environ["SLURM_JOBID"])
            run_name = "baseline-saving-" + run.info.run_name
            mlflow.set_tag("mlflow.runName", run_name)
            # mlflow.get_tag()
            print("run_name:", run_name)
            print("tracking uri:", mlflow.get_tracking_uri())
            print("artifact uri:", mlflow.get_artifact_uri())
            main(config)
    else:
        main(config)
