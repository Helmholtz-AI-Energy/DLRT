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
    # inp = torch.randn(1, 3, 10, 12)
    w = torch.randn(10, 3, 4, 5)
    # inp_unf = torch.nn.functional.unfold(inp, (4, 5))
    weight2 = w.view(w.size(0), -1)  # .T.transpose(1, 2)
    # print(weight2.shape)
    u, s, vh = torch.linalg.svd(weight2, full_matrices=False)
    print(f"u: {u.shape} s: {s.shape} vh: {vh.shape}")
    # print(f"")
    # print(f"")





    # print(f"inp shape: {inp_unf.transpose(1, 2).shape} weight2 shape: {weight2.shape}")
    # # out_unf = inp_unf.transpose(1, 2) @ weight2
    # # # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
    # # # or equivalently (and avoiding a copy),
    # # out = out_unf.view(1, 2, 7, 8)
    # layer = torch.nn.Conv2d(3, 3, kernel_size=(4, 5))
    # print(layer.weight.shape)
    # # print(torch.nn.functional.conv2d(inp, w) - out).abs().max()


def test_transformer():
    model = TransformerModel(ntoken=3, d_model=4, d_hid=5, nhead=2,
                 nlayers=4, dropout=0.5)
    print(model)


if __name__ == "__main__":
    main()
    # test_transformer()
