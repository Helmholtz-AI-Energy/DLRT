from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

from dlrt import DLRTTrainer


def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ],
    )

    batch_size = 4

    trainset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    # testset = datasets.CIFAR10(
    #     root="./data",
    #     train=False,
    #     download=True,
    #     transform=transform,
    # )
    # testloader = torch.utils.data.DataLoader(
    #     testset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=2,
    # )

    # convert model to DLRTNet
    model = models.resnet18()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # testing resnet18
    # print(model)
    dlrt_model = DLRTTrainer(
        torch_model=model,
        optimizer_name="SGD",
        optimizer_kwargs={"lr": 0.001, "momentum": 0.9},
        adaptive=True,
        loss_function=criterion,
    )
    print(dlrt_model)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            loss = dlrt_model.train_step(inputs, labels)

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")


if __name__ == "__main__":
    main()
