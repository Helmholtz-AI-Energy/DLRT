from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class DLRALinear(nn.Module):
    # overwrite the original layer depending on its type?
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        TODO: ...this

        Parameters
        ----------
        in_features
        out_features
        bias
        device
        dtype
        rank
        load_weights
        """

        rank = rank if rank is not None else min([in_features, out_features])
        if rank > in_features:
            raise ValueError(
                f"rank > in_features ({rank} > {in_features}) use nn.Linear or reduce rank",
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # def reset_parameters(self) -> None:
        #     # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        #     # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        #     # https://github.com/pytorch/pytorch/issues/57109
        #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #     if self.bias is not None:
        #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #         init.uniform_(self.bias, -bound, bound)
        #
        # def forward(self, input: Tensor) -> Tensor:
        #     return F.linear(input, self.weight, self.bias)
        #
        # def extra_repr(self) -> str:
        #     return 'in_features={}, out_features={}, bias={}'.format(
        #         self.in_features, self.out_features, self.bias is not None
        #     )

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.rank = rank

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.load_weights == None:
            self.reset_parameters()
        else:
            param, b = self.load_weights
            self.weight = torch.nn.Parameter(param)
            if bias:
                self.bias = torch.nn.Parameter(b)
            else:
                self.register_parameter("bias", None)

        if self.rank == None:
            r = min([in_features, out_features])

        elif type(self.rank) == int and self.rank <= torch.min(torch.tensor(self.weight.shape)):
            U, S, V = tuple(torch.linalg.svd(self.weight))
            U = torch.nn.Parameter(U[:, 0 : self.rank], requires_grad=False)
            S = torch.nn.Parameter(torch.diag(S[0 : self.rank]), requires_grad=False)
            V = V.T[:, 0 : self.rank]
            V = torch.nn.Parameter(V, requires_grad=False)
            # adding attributes to the weight
            setattr(self.weight, "USV", (U, S, V))

        setattr(self.weight, "lr", True)
        setattr(self.weight, "rank", self.rank)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        x = LinearFunction.apply(input, self.weight, self.bias)
        return x
