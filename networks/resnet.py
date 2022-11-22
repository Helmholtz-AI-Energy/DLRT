from __future__ import annotations

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import comm
import datasets as dsets
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

ImageFile.LOAD_TRUNCATED_IMAGES = True

import dlrt


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "--data",
    metavar="DIR",
    nargs="?",
    default="imagenet",
    help="path to dataset (default: imagenet)",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs",
    default=90,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    metavar="M",
    help="momentum",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    action="store_true",
    help="use pre-trained model",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank",
    default=-1,
    type=int,
    help="node rank for distributed training",
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend",
    default="nccl",
    type=str,
    help="distributed backend",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="seed for initializing training. ",
)
parser.add_argument("--dummy", action="store_true", help="use fake data to benchmark")

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # initialize the torch process group across all processes
    print("comm init")
    if int(os.environ["SLURM_NTASKS"]) > 1:
        comm.init(method="nccl-slurm")

    # create model
    if args.pretrained:
        print(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if dist.is_initialized():
        args.gpu = dist.get_rank() % torch.cuda.device_count()  # only 4 gpus/node
    else:
        args.gpu = 0
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay
    # )
    # print("converting model to DLRT")
    dlrt_trainer = dlrt.DLRTTrainer(
        torch_model=model,
        optimizer_name="SGD",
        optimizer_kwargs={
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
        },
        adaptive=True,
        criterion=nn.CrossEntropyLoss().to(device),
        init_ddp=dist.is_initialized(),
        mixed_precision=False,
    )
    # print(dlrt_trainer)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(dlrt_trainer.optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    # TODO: add DLRT checkpointing
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{args.gpu}"
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]),
            )
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # Data loading code
    # if args.dummy:
    #     print("=> Dummy data is used!")
    #     train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
    #     val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    # else:
    dset_dict = dsets.get_imagenet_datasets(args.data, args.batch, args.workers)
    train_loader, train_sampler = dset_dict["train"]["loader"], dset_dict["train"]["sampler"]
    val_loader = dset_dict["val"]["loader"]

    if args.evaluate:
        validate(val_loader, dlrt_trainer, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, dlrt_trainer, epoch, device, args)

        # evaluate on validation set
        _ = validate(val_loader, dlrt_trainer, args)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        # if not args.multiprocessing_distributed or args.rank % 4 == 0:
        #     save_checkpoint(
        #         {
        #             'epoch': epoch + 1,
        #             'arch': args.arch,
        #             'state_dict': model.state_dict(),
        #             'best_acc1': best_acc1,
        #             'optimizer': optimizer.state_dict(),
        #             'scheduler': scheduler.state_dict()
        #         }, is_best
        #     )


def train(train_loader, trainer: dlrt.DLRTTrainer, epoch, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    trainer.train()
    # rank = dist.get_rank()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = trainer.train_step(images, target, adapt=(epoch > 0) or (i > 100))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.output, target, topk=(1, 5))
        losses.update(output.loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i == 2:
        #    break

        if i % args.print_freq == 0:  # and rank == 0:
            progress.display(i + 1)


def validate(val_loader, trainer: dlrt.DLRTTrainer, args):
    def run_validate(loader, base_progress=0):
        rank = 0 if not dist.is_initialized() else dist.get_rank()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = trainer.valid_step(images, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.output, target, topk=(1, 5))
                losses.update(output.loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 and rank == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    trainer.eval()

    run_validate(val_loader)

    top1.all_reduce()
    top5.all_reduce()

    if len(val_loader.sampler) * args.world_size < len(val_loader.dataset):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    top1.all_reduce()
    top5.all_reduce()

    # if dist.get_rank() == 0:
    #    print(f"End of epoch, validation top1: {top1}\ttop5: {top5}")

    return top1.avg


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
        self.avg = self.sum / self.count

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
        if self.rank == 0:
            print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if self.rank == 0:
            print(" ".join(entries))

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
    main()
