from __future__ import annotations

from pathlib import Path

import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as datadist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timm.data.transforms_factory import create_transform

# from timm.data import

imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

cifar10_normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)
mnist_normalize = transforms.Normalize(
    mean=(0.1307,),
    std=(0.3081,),
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


def get_cifar100_datasets(base_dir, batch_size, workers):
    train_dataset, train_loader, train_sampler = cifar100_train_dataset_plus_loader(
        base_dir=base_dir,
        batch_size=batch_size,
        workers=workers,
    )
    val_dataset, val_loader = cifar100_val_dataset_n_loader(
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


def get_mnist_datasets(base_dir, batch_size, workers, resize):
    train_dataset, train_loader, train_sampler = mnist_train_data(
        base_dir=base_dir,
        batch_size=batch_size,
        workers=workers,
        resize=resize,
    )
    val_dataset, val_loader = mnist_val_data(
        base_dir=base_dir,
        batch_size=batch_size,
        workers=workers,
        resize=resize,
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
                transforms.RandomResizedCrop(176),
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
                transforms.CenterCrop(232),
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

    # if dist.is_initialized():
    #     sampler = datadist.DistributedSampler(test_dataset)
    # else:
    sampler = None

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        sampler=sampler,
    )
    return test_dataset, test_loader


def cifar100_train_dataset_plus_loader(base_dir, batch_size, workers=6):
    # CIFAR-10 dataset
    train_dir = Path(base_dir) / "train"

    timm_transforms = create_transform(
        32,
        is_training=True,
        auto_augment="rand-m9-mstd0.5",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )

    # transform = transforms.Compose(
    #     [
    #         transforms.Pad(4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32),
    #         transforms.ToTensor(),
    #         cifar10_normalize,
    #     ],
    # )

    train_dataset = datasets.CIFAR100(
        root=str(train_dir),
        train=True,
        transform=timm_transforms,  # transform,  # timm_transforms,
        # download=True,
    )

    # Data loader
    if dist.is_initialized():
        train_sampler = datadist.DistributedSampler(train_dataset, shuffle=True)
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


def cifar100_val_dataset_n_loader(base_dir, batch_size, workers=6):
    val_dir = Path(base_dir) / "val"

    test_dataset = datasets.CIFAR100(
        root=str(val_dir),
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), cifar10_normalize]),
        # download=True,
    )

    if dist.is_initialized():
        sampler = datadist.DistributedSampler(test_dataset, shuffle=True)
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


def mnist_train_data(base_dir, batch_size, workers=2, resize=False):
    if resize:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=32),
                transforms.Grayscale(3),
                mnist_normalize,
            ],
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), mnist_normalize])
    train_dataset = datasets.MNIST(base_dir, train=True, download=True, transform=transform)

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


def mnist_val_data(base_dir, batch_size, workers=2, resize=False):
    if resize:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=32),
                transforms.Grayscale(3),
                mnist_normalize,
            ],
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), mnist_normalize])
    val_dataset = datasets.MNIST(base_dir, train=False, transform=transform)
    sampler = None

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        sampler=sampler,
    )
    return val_dataset, val_loader
