from __future__ import annotations

import collections
import math
from itertools import repeat
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from rich.columns import Columns
from rich.console import Console
from rich.pretty import Pretty
from torch import Tensor
from torch.nn import common_types

from .basic import DLRTModule

console = Console(width=140)

__all__ = ["DLRTConv2d", "DLRTConv2dAdaptive", "DLRTConv2dFixed"]


def DLRTConv2d(
    adaptive: bool,
    in_channels: int,
    out_channels: int,
    kernel_size: common_types._size_2_t,
    stride: common_types._size_2_t = 1,
    padding: str | common_types._size_2_t = 0,
    dilation: common_types._size_2_t = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    dtype=None,
    device=None,
    low_rank_percent=None,
    eps_adapt: float = 0.01,
    convert_from_weights: Tensor = None,
    existing_bias: Tensor = None,
    pretrain: bool = True,
):
    if adaptive:
        return DLRTConv2dAdaptive(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            dtype,
            device,
            low_rank_percent,
            eps_adapt,
            # convert_from_weights=convert_from_weights,
            # existing_bias=existing_bias,
            pretrain=pretrain,
        )
    else:
        return DLRTConv2dFixed(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            dtype,
            device,
            low_rank_percent,
            convert_from_weights=convert_from_weights,
            existing_bias=existing_bias,
            # pretrain=pretrain,  TODO
        )


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")


class _ConvNd(DLRTModule):
    # Taken directly from torch
    # (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py)

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    # todo: decide on if this should be used. only to match up with torch
    # def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    #     ...

    _in_channels: int
    _reversed_padding_repeated_twice: list[int]
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Tensor | None
    # ==== low_rank =================================================
    low_rank: int
    rmax: int
    convert_from_weights: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
        # ====================== DLRT params =========================================
        low_rank_percent: float | None = None,
        fixed_rank: bool = False,
        convert_from_weights: torch.Tensor = None,
        # existing_bias: Tensor = None,
        pretrain: bool = True,
    ) -> None:
        # from torch =================================================================
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding,
                        valid_padding_strings,
                    ),
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes,
                    padding_mode,
                ),
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation,
                    kernel_size,
                    range(len(kernel_size) - 1, -1, -1),
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad
        else:
            self._reversed_padding_repeated_twice = nn.modules.utils._reverse_repeat_tuple(self.padding, 2)

        # new changes =================================================================
        self.fixed_rank = fixed_rank
        kernel_size_number = np.prod(self.kernel_size)
        self.kernel_size_number = kernel_size_number

        self.basic_number_weights = in_channels * (out_channels // groups + kernel_size_number)
        # a, b = in_channels, (out_channels // groups + kernel_size_number)

        # TODO: fix me!
        self.low_rank = int(
            min([self.out_channels, self.in_channels * self.kernel_size_number]) / 2,
        )
        self.rmax = self.low_rank * 2
        # print("rmax", self.rmax, self.out_channels, self.in_channels * self.kernel_size_number)

        # # TODO: fix me???? not sure if this is an issue here or not
        # if low_rank_percent is None:
        #     # set the max low_rank to be such that the
        #     roots = np.roots([1, a + b, a * b])
        #     pos_coeff = roots[roots > 0]  # TODO: adjust me?
        #     if len(pos_coeff) < 1:
        #         self.rmax = min([a, b]) // 2
        #     else:
        #         self.rmax = int(np.floor(pos_coeff[-1]))
        #     # set the initial low_rank to be most of the rmax
        #     if self.rmax < 10:
        #         self.rmax = 20
        #     self.low_rank = self.rmax // 2
        #     # print("rmax: ", self.rmax, "low_rank: ", self.low_rank)
        # else:
        #     self.rmax = min([a, b]) // 2
        #     self.low_rank = int(self.rmax * low_rank_percent * 10)
        #     self.rmax = int(self.low_rank * 2)  # TODO: cleanup
        #
        # if self.low_rank == 0:
        #     print(a, b, out_channels)
        #     self.low_rank = 100
        #     self.rmax = int(self.low_rank * 2)

        # TODO: transposed things

        # ======= from torch =================================
        # if transposed:
        #     self.weight = nn.Parameter(
        #         torch.empty(
        #             (in_channels, out_channels // groups, *kernel_size), **factory_kwargs
        #         )
        #     )
        # else:
        #     self.weight = nn.Parameter(
        #         torch.empty(
        #             (out_channels, in_channels // groups, *kernel_size), **factory_kwargs
        #         )
        #     )

        # if existing_bias is not None:
        #     self.bias = existing_bias.clone()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, requires_grad=False, **factory_kwargs),
                requires_grad=False,
            )

    def extra_repr(self):
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += f", output_padding={self.output_padding}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode}"
        s += f", low_rank={self.low_rank}"
        # return s.format(**self.__dict__)
        return s

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class DLRTConv2dFixed(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: common_types._size_2_t,
        stride: common_types._size_2_t = 1,
        padding: str | common_types._size_2_t = 0,
        dilation: common_types._size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype=None,
        device=None,
        low_rank_percent=None,
        convert_from_weights: torch.Tensor = None,
        existing_bias: Tensor = None,
    ) -> None:
        """
        Initializer for the convolutional low rank layer (filterwise), extention of the classical Pytorch's convolutional layer.
        INPUTS:
        in_channels: number of input channels (Pytorch's standard)
        out_channels: number of output channels (Pytorch's standard)
        kernel_size : kernel_size for the convolutional filter (Pytorch's standard)
        dilation : dilation of the convolution (Pytorch's standard)
        padding : padding of the convolution (Pytorch's standard)
        stride : stride of the filter (Pytorch's standard)
        bias  : flag variable for the bias to be included (Pytorch's standard)
        step : string variable ('K','L' or 'S') for which forward phase to use
        rank : rank variable, None if the layer has to be treated as a classical Pytorch Linear layer (with weight and bias). If
                it is an int then it's either the starting rank for adaptive or the fixed rank for the layer.
        fixed : flag variable, True if the rank has to be fixed (KLS training on this layer)
        load_weights : variables to load (Pytorch standard, to finish)
        dtype : Type of the tensors (Pytorch standard, to finish)
        """
        # TODO: fix init
        #   TODO: maybe remove this and simply use adaptive instead but just dont call the adapt
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=False,
            output_padding=_pair(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            # ====================== DLRT params =========================================
            low_rank_percent=low_rank_percent,
            fixed_rank=False,
        )

        self.train_case = "k"

        if self.bias is not None:
            self.bias = torch.nn.Parameter(
                torch.zeros(self.out_channels, requires_grad=False, **factory_kwargs),
                requires_grad=False,
            )
        # n, m = self.out_channels, self.in_channels * self.kernel_size_number
        in_kern = self.in_channels * self.kernel_size_number
        self.u = nn.Parameter(
            torch.empty((self.out_channels, self.low_rank), **factory_kwargs),
            requires_grad=False,
        )
        self.u_hat = nn.Parameter(
            torch.empty((self.out_channels, self.low_rank), **factory_kwargs),
            requires_grad=False,
        )
        self.v = nn.Parameter(
            torch.empty((in_kern, self.low_rank), **factory_kwargs),
            requires_grad=False,
        )
        self.v_hat = nn.Parameter(
            torch.empty((in_kern, self.low_rank), **factory_kwargs),
            requires_grad=False,
        )
        self.s_hat = nn.Parameter(
            torch.empty((self.low_rank, self.low_rank), **factory_kwargs),
            requires_grad=True,
        )
        self.k = nn.Parameter(
            torch.empty((self.out_channels, self.low_rank), **factory_kwargs),
            requires_grad=True,
        )
        self.l = nn.Parameter(  # noqa: E741
            torch.empty((in_kern, self.low_rank), **factory_kwargs),
            requires_grad=True,
        )
        self.n_hat = torch.nn.Parameter(
            torch.empty((self.low_rank, self.low_rank), **factory_kwargs),
            requires_grad=False,
        )
        self.m_hat = torch.nn.Parameter(
            torch.empty((self.low_rank, self.low_rank), **factory_kwargs),
            requires_grad=False,
        )
        # todo: convert from full rank
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.s_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.u_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.l, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.n_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.m_hat, a=math.sqrt(5))

        if self.bias is not None:
            weight = torch.empty(
                (self.out_channels, self.in_channels // self.groups, *self.kernel_size),
                device=self.k.device,
                dtype=self.k.dtype,
            )
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            del weight

    def forward(self, input):
        """
        forward phase for the convolutional layer. It has to contain the three different
        phases for the steps 'K','L' and 'S' in order to be optimizable using dlrt.

        """

        # batch_size = input.shape[0]
        batch_size, shp1, shp2, shp3 = tuple(input.shape)
        pad0, dil0, kern0 = self.padding[0], self.dilation[0], self.kernel_size[0]
        pad1, dil1, kern1 = self.padding[1], self.dilation[1], self.kernel_size[1]
        stride0 = self.stride[0]
        stride1 = self.stride[1]
        out_h = int((shp2 + 2 * pad0 - dil0 * (kern0 - 1) - 1) // stride0) + 1
        out_w = int((shp3 + 2 * pad1 - dil1 * (kern1 - 1) - 1) // stride1) + 1

        # out_h = int(np.floor(((input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (
        #                 self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1))
        # out_w = int(np.floor(((input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (
        #                   self.kernel_size[1] - 1) - 1) / self.stride[1]) + 1))

        inp_unf = (
            F.unfold(
                input,
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
            )
            .to(input.device)
            .transpose(1, 2)
        )
        if self.train_case == "k":
            # print(inp_unf.shape, self.v.shape, self.k.T.shape)
            out_unf = inp_unf @ self.v @ self.k.T
            # out_unf = torch.linalg.multi_dot([inp_unf.transpose(1, 2), self.v, self.k.T])
        elif self.train_case == "l":
            out_unf = inp_unf @ self.l @ self.u.T
            # out_unf = torch.linalg.multi_dot([inp_unf.transpose(1, 2), self.l, self.u.T])
        elif self.train_case == "s":
            out_unf = inp_unf @ torch.linalg.multi_dot(
                [self.v, self.s_hat.T, self.u.T],
                # [self.v, torch.diag(self.s_hat), self.u.T],
            )
        else:
            raise ValueError(f"Invalude step value: {self.step}")

        if self.bias is not None:
            out_unf.add_(self.bias)
        # else:
        out_unf.transpose_(1, 2)
        return out_unf.view(batch_size, self.out_channels, out_h, out_w)

    def _change_params_requires_grad(self, requires_grad):
        self.k.requires_grad = requires_grad
        self.s_hat.requires_grad = requires_grad
        self.l.requires_grad = requires_grad
        self.u.requires_grad = False  # requires_grad
        # self.u_hat.requires_grad = False  # requires_grad
        self.v.requires_grad = False  # requires_grad
        # self.v_hat.requires_grad = False  # requires_grad
        self.n_hat.requires_grad = False  # requires_grad
        self.m_hat.requires_grad = False  # requires_grad
        if self.bias is not None:
            self.bias.requires_grad = requires_grad

    @torch.no_grad()
    def k_preprocess(self):
        self._change_params_requires_grad(False)
        self.k.set_(self.u @ self.s_hat)
        self.k.requires_grad = True

    @torch.no_grad()
    def k_postprocess(self):
        u_hat, _ = torch.linalg.qr(self.k)
        # TESTING: setting M and U with 'undoing' the k preprocess
        self.m_hat.set_(u_hat.T @ self.u)
        self.u.set_(u_hat)

    @torch.no_grad()
    def l_preprocess(self):
        self._change_params_requires_grad(False)
        self.l.set_(self.v @ self.s_hat.T)
        self.l.requires_grad = True

    @torch.no_grad()
    def l_postprocess(self):
        v_hat, _ = torch.linalg.qr(self.l)
        self.n_hat.set_(v_hat.T @ self.v)
        self.v.data = v_hat

    @torch.no_grad()
    def s_preprocess(self):
        self._change_params_requires_grad(False)
        self.s_hat.set_(torch.linalg.multi_dot([self.m_hat, self.s_hat, self.n_hat.T]))
        self.s_hat.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True


class DLRTConv2dAdaptive(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: common_types._size_2_t,
        stride: common_types._size_2_t = 1,
        padding: str | common_types._size_2_t = 0,
        dilation: common_types._size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype=None,
        device=None,
        low_rank_percent=None,
        eps_adapt: float = 0.01,
        pretrain: bool = True,
    ) -> None:
        """
        Initializer for the convolutional low rank layer (filterwise), extention of the classical Pytorch's convolutional layer.
        INPUTS:
        in_channels: number of input channels (Pytorch's standard)
        out_channels: number of output channels (Pytorch's standard)
        kernel_size : kernel_size for the convolutional filter (Pytorch's standard)
        dilation : dilation of the convolution (Pytorch's standard)
        padding : padding of the convolution (Pytorch's standard)
        stride : stride of the filter (Pytorch's standard)
        bias  : flag variable for the bias to be included (Pytorch's standard)
        step : string variable ('K','L' or 'S') for which forward phase to use
        rank : rank variable, None if the layer has to be treated as a classical Pytorch Linear layer (with weight and bias). If
                it is an int then it's either the starting rank for adaptive or the fixed rank for the layer.
        load_weights : variables to load (Pytorch standard, to finish)
        dtype : Type of the tensors (Pytorch standard, to finish)
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=False,
            output_padding=_pair(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            # ====================== DLRT params =========================================
            low_rank_percent=low_rank_percent,
            fixed_rank=False,
            pretrain=pretrain,
        )
        self.train_case = "k"
        self.eps_adapt = eps_adapt

        in_kern = self.in_channels * self.kernel_size_number
        self.in_kern = in_kern
        self.pretrain = pretrain
        if pretrain:
            self.fullweight = nn.Parameter(
                torch.empty(out_channels, in_kern),
                requires_grad=True,
            )
        # ONLY create the parameters, reset_parameters fills them
        self.s_hat = nn.Parameter(
            torch.empty(
                self.rmax,
                self.rmax,
                **factory_kwargs,
            ),
            requires_grad=False,
        )
        self.u = nn.Parameter(
            torch.empty(self.out_channels, self.rmax, **factory_kwargs),
            requires_grad=False,
        )
        self.u_hat = nn.Parameter(
            torch.empty(self.out_channels, 2 * self.rmax, **factory_kwargs),
            requires_grad=False,
        )
        self.v = nn.Parameter(
            torch.empty(in_kern, self.rmax, **factory_kwargs),
            requires_grad=False,
        )
        self.v_hat = nn.Parameter(
            torch.empty(in_kern, 2 * self.rmax, **factory_kwargs),
            requires_grad=False,
        )
        self.k = nn.Parameter(
            torch.empty(self.out_channels, self.rmax, **factory_kwargs),
            requires_grad=False,
        )
        self.l = torch.nn.Parameter(  # noqa: E741
            torch.empty(in_kern, self.rmax, **factory_kwargs),
            requires_grad=False,
        )
        self.n_hat = torch.nn.Parameter(
            torch.empty(2 * self.rmax, self.rmax, **factory_kwargs),
            requires_grad=False,
        )
        self.m_hat = torch.nn.Parameter(
            torch.empty(2 * self.rmax, self.rmax, **factory_kwargs),
            requires_grad=False,
        )
        self.reset_parameters()

        # self.existing_bias = existing_bias is not None
        # if convert_from_weights is not None:
        #     self.convert_from_full_rank(
        #         weight=convert_from_weights,
        #         starting_rank=None, #low_rank_percent,  # self.low_rank,
        #     )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # # below is with normal initialization?
        if self.pretrain:
            nn.init.kaiming_uniform_(self.fullweight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.s_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.u_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.l, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.n_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.m_hat, a=math.sqrt(5))

        # for testing
        if self.bias is not None:  # and not self.existing_bias:
            weight = torch.empty(
                (self.out_channels, self.in_channels // self.groups, *self.kernel_size),
                device=self.k.device,
                dtype=self.k.dtype,
            )
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            del weight

    def train(self, mode=True):
        self.training = mode

    def forward(self, input: Tensor) -> Tensor:
        """
        forward phase for the convolutional layer. It has to contain the three different
        phases for the steps 'K','L' and 'S' in order to be optimizable using dlrt.

        """
        # batch_size = input.shape[0]
        batch_size, shp1, shp2, shp3 = tuple(input.shape)
        pad0, dil0, kern0 = self.padding[0], self.dilation[0], self.kernel_size[0]
        pad1, dil1, kern1 = self.padding[1], self.dilation[1], self.kernel_size[1]
        stride0 = self.stride[0]
        stride1 = self.stride[1]
        out_h = int((shp2 + 2 * pad0 - dil0 * (kern0 - 1) - 1) // stride0) + 1
        out_w = int((shp3 + 2 * pad1 - dil1 * (kern1 - 1) - 1) // stride1) + 1
        # out_h = int(np.floor(((input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (
        #                 self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1))
        # out_w = int(np.floor(((input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (
        #                   self.kernel_size[1] - 1) - 1) / self.stride[1]) + 1))

        inp_unf = (
            F.unfold(
                input,
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
            )
            .to(input.device)
            .transpose(1, 2)
        )
        eps = torch.finfo(inp_unf.dtype).eps  # noqa: F841
        if self.train_case == "pretrain":
            out_unf = inp_unf @ self.fullweight.T
        elif self.train_case == "k" or not self.training:
            # TODO: fastest method: (inp_unf @ v) @ k.T
            k, v = self.k[:, : self.low_rank].T, self.v[:, : self.low_rank]
            # second = v @ k
            # second[(second <= eps) & (second >= -eps)] *= 0
            # out_unf = inp_unf @ second  # @ v @ k
            # Transposed k in line above
            out_unf = (inp_unf @ v) @ k
        elif self.train_case == "l" and self.training:
            l, ut = self.l[:, : self.low_rank], self.u[:, : self.low_rank].T
            # second = l @ ut
            # second[(second <= eps) & (second >= -eps)] *= 0
            # out_unf = inp_unf @ second
            out_unf = (inp_unf @ l) @ ut
        elif self.train_case == "s" or not self.training:
            # TODO: is this set to 2*lr to enable the ability to grow lr?
            u_hat = self.u_hat[:, : 2 * self.low_rank]
            s_hat = self.s_hat[: 2 * self.low_rank, : 2 * self.low_rank]
            v_hat = self.v_hat[:, : 2 * self.low_rank]
            # fastest method uses multi_dot A x (B X (C x D))
            second = torch.linalg.multi_dot([v_hat, s_hat.T, u_hat.T])
            # second[(second <= eps) & (second >= -eps)] *= 0
            out_unf = inp_unf @ second
        else:
            raise ValueError(f"Pretraining? {self.pretrain}...Invalid step value: {self.train_case}")

        if self.bias is not None:
            out_unf.add_(self.bias)

        # undo transpose
        out_unf.transpose_(1, 2)
        return out_unf.view(batch_size, self.out_channels, out_h, out_w)

    def set_dlrt_requires_grad(self, requires):
        self.k.requires_grad = requires
        self.s_hat.requires_grad = requires
        self.l.requires_grad = requires
        self.u.requires_grad = False  # requires_grad
        self.u_hat.requires_grad = False  # requires_grad
        self.v.requires_grad = False  # requires_grad
        self.v_hat.requires_grad = False  # requires_grad
        self.n_hat.requires_grad = False  # requires_grad
        self.m_hat.requires_grad = False  # requires_grad

    def _change_params_requires_grad(self, requires_grad):
        self.set_dlrt_requires_grad(requires_grad)
        self.bias.requires_grad = requires_grad

    @torch.no_grad()
    def k_preprocess(self):
        self._change_params_requires_grad(False)
        # k prepro
        # k -> aux_U @ s
        k = self.u[:, : self.low_rank] @ self.s_hat[: self.low_rank, : self.low_rank]
        # self.k.zero_()
        self.k[:, : self.low_rank] = k
        self.k.requires_grad = True

    @torch.no_grad()
    def l_preprocess(self):
        self._change_params_requires_grad(False)
        L = self.v[:, : self.low_rank] @ self.s_hat[: self.low_rank, : self.low_rank].T
        # self.l.zero_()
        self.l[:, : self.low_rank] = L
        self.l.requires_grad = True

    @torch.no_grad()
    def k_postprocess(self):
        lr = self.low_rank
        lr2 = 2 * self.low_rank
        u_hat, _ = torch.linalg.qr(torch.hstack((self.k[:, :lr], self.u[:, :lr])))
        # self.u_hat.zero_()
        # self.m_hat.zero_()
        self.u_hat[:, :lr2] = u_hat
        self.m_hat[:lr2, :lr] = self.u_hat[:, :lr2].T @ self.u[:, :lr]

    @torch.no_grad()
    def l_postprocess(self):
        lr = self.low_rank
        lr2 = 2 * self.low_rank
        v_hat, _ = torch.linalg.qr(torch.hstack((self.l[:, :lr], self.v[:, :lr])))
        # self.v_hat.zero_()
        # self.n_hat.zero_()
        self.v_hat[:, :lr2] = v_hat
        self.n_hat[:lr2, :lr] = self.v_hat[:, :lr2].T @ self.v[:, :lr]

    @torch.no_grad()
    def s_preprocess(self):
        self._change_params_requires_grad(False)

        lr = self.low_rank
        lr2 = 2 * self.low_rank

        s = torch.linalg.multi_dot(
            [
                self.m_hat[:lr2, :lr],
                self.s_hat[:lr, :lr],
                self.n_hat[: 2 * lr, :lr].T,
            ],
        )
        # self.s_hat.zero_()
        self.s_hat[:lr2, :lr2] = s

        # bias is trainable for the s step
        # set s -> (aux_N @ s) @ aux_M.T
        self.s_hat.requires_grad = True
        self.bias.requires_grad = True

    @torch.no_grad()
    def rank_adaption(self, skip=False):
        # 1) compute SVD of S
        # d=singular values, u2 = left singuar vecs, v2= right singular vecs
        # TODO: 64 bit?
        s_small = self.s_hat[: 2 * self.low_rank, : 2 * self.low_rank]  # .clone().detach()
        try:
            u2, sing, vh2 = torch.linalg.svd(
                s_small,
                full_matrices=False,
                # driver="gesvdj",
            )
        except torch._C._LinAlgError as e:
            print(f"LinAlgError during SVD -> {e}")
            return
        v2 = vh2.T.to(self.s_hat.dtype, non_blocking=True)
        u2 = u2.to(self.s_hat.dtype, non_blocking=True)

        # absolute value treshold (try also relative one)
        # TODO: different threshold methods

        if not skip:
            tol = self.eps_adapt * torch.linalg.norm(sing)
            new_lr = sing.shape[0] // 2
            for j in range(0, 2 * new_lr - 1):
                tmp = torch.linalg.norm(sing[j : 2 * new_lr - 1])
                if tmp < tol:
                    new_lr = j
                    break
        else:
            new_lr = self.low_rank

        # new_lr = max(min(new_lr, self.rmax), 2)  # -> handled by the 2 in the for loop above
        # update s

        # new lr should only be a minimum of 2! (but it shouldn't really ever get here....)
        new_lr = max([min([new_lr, self.low_rank]), 2])
        if new_lr > 2 * self.low_rank:
            print("new lr > 2*old lr!!")
            # self.u.set_(self.u_hat.data)
            # self.v.set_(self.v_hat.data)
            # return
            # todo: raise??
        # self.s_hat.set_(
        #     torch.eye(*self.s_hat.shape, device=self.s_hat.device, dtype=self.s_hat.dtype)
        # )
        # self.s_hat.zero_()
        # self.u.zero_()
        # self.v.zero_()

        # lst = []
        # u, s, v = None, None, None
        # for n, p in dlrt_trainer.dlrt_model.named_parameters():
        #     # if n.endswith("s_hat") or n.endswith("u") or n.endswith("v"):
        #     #     try:
        #     #         lst.append(f'{n}: {p.mean():.4f} {p.min():.4f} {p.max():.4f} {p.std():.4f}')
        #     #     except:
        #     #         pass
        #     if n == "torch_model.conv1.s_hat":
        #         s = p
        #     elif n == "torch_model.conv1.u":
        #         u = p
        #     elif n == "torch_model.conv1.v":
        #         v = p
        # fwr = u @ s @ v.T  # full weight representation
        # time.sleep(config['rank'] * 2)
        # # cols = Columns(lst, equal=True, expand=True)
        # # rprint(fwr.shape)
        # mxsz = max(tuple(fwr.shape))
        # loc_fwrep = torch.eye(mxsz).to(device=fwr.device)
        # loc_fwrep[:fwr.shape[0], :fwr.shape[1]] = fwr
        # w0 = torch.zeros_like(loc_fwrep)
        # if dist.get_rank() == 0:
        #     w0 = loc_fwrep
        # dist.broadcast(w0, src=0)

        self.s_hat[:new_lr, :new_lr] = torch.diag(sing[:new_lr]).to(
            device=self.s_hat.device,
            dtype=self.s_hat.dtype,
        )
        self.u[:, :new_lr] = self.u_hat[:, : 2 * self.low_rank] @ u2[:, :new_lr]
        self.v[:, :new_lr] = self.v_hat[:, : 2 * self.low_rank] @ v2[:, :new_lr]

        self.low_rank = int(new_lr)

    @torch.no_grad()
    def all_reduce(self, method: str = "average"):
        if not dist.is_initialized():
            # early out if not working distributed
            return
        #   (full weight and bias are changed 'as expected' for a nn
        # this has no effect if in pretraining

        # TODO: groups?
        sz = float(dist.get_world_size())
        if method == "average":
            # if dist.get_rank() == 0:
            # console.rule("u")
            # s = ""
            # for d in torch.diag(self.u)[:10].tolist():
            #     s += f" {d:.3f}"
            # console.print(s)
            # print(self.u)
            # raise ValueError

            self.u.true_divide_(sz)
            self.s_hat.true_divide_(sz)
            self.v.true_divide_(sz)
            self.u_hat.true_divide_(sz)
            self.v_hat.true_divide_(sz)
            self.k.true_divide_(sz)
            self.l.true_divide_(sz)
            self.n_hat.true_divide_(sz)
            self.m_hat.true_divide_(sz)
            dist.all_reduce(self.u, op=dist.ReduceOp.SUM, async_op=True)
            dist.all_reduce(self.s_hat, op=dist.ReduceOp.SUM, async_op=True)
            dist.all_reduce(self.v, op=dist.ReduceOp.SUM, async_op=True)
            dist.all_reduce(self.u_hat, op=dist.ReduceOp.SUM, async_op=True)
            dist.all_reduce(self.v_hat, op=dist.ReduceOp.SUM, async_op=True)
            dist.all_reduce(self.k, op=dist.ReduceOp.SUM, async_op=True)
            dist.all_reduce(self.l, op=dist.ReduceOp.SUM, async_op=True)
            dist.all_reduce(self.n_hat, op=dist.ReduceOp.SUM, async_op=True)
            dist.all_reduce(self.m_hat, op=dist.ReduceOp.SUM, async_op=True)
            # if dist.get_rank() == 0:
            #     console.print(torch.diag(self.s_hat)[:10].tolist())
        elif method == "projection":
            # TODO: transpose V?
            # 1. get full weight representation
            fwr = self.u @ self.s_hat @ self.v.T  # full weight representation
            # 2. pad weight representation into
            mxsz = max(tuple(fwr.shape))
            loc_fwrep = torch.eye(mxsz).to(device=fwr.device)
            loc_fwrep[: fwr.shape[0], : fwr.shape[1]] = fwr
            w0 = torch.zeros_like(loc_fwrep)
            if dist.get_rank() == 0:
                w0 = loc_fwrep
            dist.broadcast(w0, src=0)
        else:
            raise ValueError(f"invalid all reduce method: {method}")

    @torch.no_grad()
    def stop_pretraining(self):
        # TODO: need to sync up the ranks in DDP!!
        self.pretrain = False

        # factory = {"dtype": weight.dtype, "device": weight.device}
        # self.to(**factory)
        # fullweight: out x in_kern -> .T : in_kern x out
        u, sing, vh = torch.linalg.svd(
            self.fullweight.T,
            full_matrices=True,  # FIXME?
            # driver="gesvdj",
        )
        # u : in_kern x in_kern
        # sing: min(out x in_kern)
        # vh: out x out
        # print(u.shape, sing.shape, vh.shape, self.v.shape)
        # u = u.to(**factory, non_blocking=True)
        # sing = sing.to(**factory, non_blocking=True)
        # v = vh.T

        new_lr = min(sing.shape[0], self.s_hat.shape[0])

        # new
        nn.init.eye_(self.s_hat)
        self.s_hat[:new_lr, :new_lr] = torch.diag(sing[:new_lr]).to(
            device=self.s_hat.device,
            dtype=self.s_hat.dtype,
        )

        # u: output x rank
        # v: in_kern x rank
        self.u[:, :new_lr] = vh[:, :new_lr]
        self.u_hat[:, :new_lr] = vh[:, :new_lr]
        self.v[:, :new_lr] = u[:, :new_lr]
        self.v_hat[:, :new_lr] = u[:, :new_lr]

        # self.u[:, :new_lr] = self.u_hat[:, : 2 * self.low_rank] @ u[:, :new_lr]
        # self.v[:, :new_lr] = self.v_hat[:, : 2 * self.low_rank] @ vh[:, :new_lr]
        # print(self.fullweight.shape, self.u.shape, self.v.shape, new_lr, self.low_rank,
        #     self.v_hat.shape)
        # self.u[:, :new_lr] = self.u_hat[:, : 2 * self.low_rank] @ u[:, :new_lr]
        # self.v[:, :new_lr] = self.v_hat[:, : 2 * self.low_rank] @ vh.T[:, :new_lr]

        del self.fullweight
        # self.low_rank = int(new_lr)
