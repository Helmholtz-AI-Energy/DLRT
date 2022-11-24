from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

from dlrt import DLRTTrainer
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


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
    print(model)
    print([c for c in model.children()][-1])
    # dlrt_model = DLRTTrainer(
    #     torch_model=model,
    #     optimizer_name="SGD",
    #     optimizer_kwargs={"lr": 0.001, "momentum": 0.9},
    #     adaptive=True,
    #     criterion=criterion,
    # )
    # print(dlrt_model)


    # for epoch in range(2):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         # optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         loss = dlrt_model.train_step(inputs, labels)
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:  # print every 2000 mini-batches
    #             print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
    #             running_loss = 0.0

    print("Finished Training")


def test_transformer():
    model = TransformerModel(ntoken=3, d_model=4, d_hid=5, nhead=2,
                 nlayers=4, dropout=0.5)
    print(model)


if __name__ == "__main__":
    main()
    # test_transformer()
