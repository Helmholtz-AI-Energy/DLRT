from __future__ import annotations

from pathlib import Path

import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as datadist
import torchvision.datasets as datasets
import torchvision.transforms as transforms

imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

cifar10_normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)


# def get_dataset(conf):
#     # get datasets
#     if conf["dataset_name"] == "imagenet":
#         train_dataset, train_loader, train_sampler = imagenet_train_dataset_plus_loader(conf)
#         val_dataset, val_loader = imagenet_get_val_dataset_n_loader(conf)
#     elif conf["dataset_name"] == "cifar10":
#         train_dataset, train_loader, train_sampler = cifar10_train_dataset_n_loader(conf)
#         val_dataset, val_loader = cifar10_val_set_n_loader(conf)
#     else:
#         raise NotImplementedError(f"'{conf['dataset_name']}' not a valid dataset name")
#
#     return {
#         "train": {
#             "dataset": train_dataset,
#             "loader": train_loader,
#             "sampler": train_sampler,
#         },
#         "val": {
#             "dataset": val_dataset,
#             "loader": val_loader,
#         },
#     }


def get_imagenet_datasets(base_dir, batch_size, workers):
    train_dataset, train_loader, train_sampler = imagenet_train_dataset_plus_loader(
        base_dir=base_dir,
        batch_size=batch_size,
        workers=workers,
    )
    val_dataset, val_loader = imagenet_get_val_dataset_n_loader(
        base_dir=base_dir,
        batch_size=batch_size,
        workers=workers,
    )
    return {
        "train": {
            "dataset": train_dataset,
            "loader": train_loader,
            "sampler": train_sampler,
        },
        "val": {
            "dataset": val_dataset,
            "loader": val_loader,
        },
    }


def get_cifar10_datasets(base_dir, batch_size, workers):
    train_dataset, train_loader, train_sampler = cifar10_train_dataset_plus_loader(
        base_dir=base_dir,
        batch_size=batch_size,
        workers=workers,
    )
    val_dataset, val_loader = cifar10_val_dataset_n_loader(
        base_dir=base_dir,
        batch_size=batch_size,
        workers=workers,
    )
    return {
        "train": {
            "dataset": train_dataset,
            "loader": train_loader,
            "sampler": train_sampler,
        },
        "val": {
            "dataset": val_dataset,
            "loader": val_loader,
        },
    }


def imagenet_train_dataset_plus_loader(base_dir, batch_size, workers=6):
    train_dir = Path(base_dir) / "train"
    train_dataset = datasets.ImageFolder(
        str(train_dir),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalize,
            ],
        ),
    )

    if dist.is_initialized():
        train_sampler = datadist.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
    )

    return train_dataset, train_loader, train_sampler


def imagenet_get_val_dataset_n_loader(base_dir, batch_size, workers=6):
    val_dir = Path(base_dir) / "val"
    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_normalize,
            ],
        ),
    )
    if dist.is_initialized():
        val_sampler = datadist.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True,
    )

    return val_dataset, val_loader


def cifar10_train_dataset_plus_loader(base_dir, batch_size, workers=6):
    # CIFAR-10 dataset
    train_dir = Path(base_dir) / "train"
    transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            cifar10_normalize,
        ],
    )

    train_dataset = datasets.CIFAR10(
        root=str(train_dir),
        train=True,
        transform=transform,
        download=True,
    )

    # Data loader
    if dist.is_initialized():
        train_sampler = datadist.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        persistent_workers=True,
    )
    return train_dataset, train_loader, train_sampler


def cifar10_val_dataset_n_loader(base_dir, batch_size, workers=6):
    val_dir = Path(base_dir) / "val"

    test_dataset = datasets.CIFAR10(
        root=str(val_dir),
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), cifar10_normalize]),
    )

    if dist.is_initialized():
        sampler = datadist.DistributedSampler(test_dataset)
    else:
        sampler = None

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        sampler=sampler,
    )
    return test_dataset, test_loader
