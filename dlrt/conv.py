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
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import common_types

from .basic import DLRTModule


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
    eps_adapt: float = 0.1,
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
        a, b = in_channels, (out_channels // groups + kernel_size_number)

        # TODO: taken from Linear -> needs to be adjusted for conv?? TESTME
        if low_rank_percent is None:
            # set the max low_rank to be such that the
            roots = np.roots([1, a + b, a * b])
            pos_coeff = roots[roots > 0]
            if len(pos_coeff) < 1:
                self.rmax = min([a, b]) // 2
            else:
                self.rmax = int(np.floor(pos_coeff[-1]))
            # set the initial low_rank to be most of the rmax
            if self.rmax < 10:
                self.rmax = 20
            self.low_rank = self.rmax // 2
            print("rmax: ", self.rmax, "low_rank: ", self.low_rank)
        else:
            self.rmax = min([a, b]) // 2
            self.low_rank = int(self.rmax * low_rank_percent)
        # self.lr = int(low_rank_percent * weight.numel())

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
        # Weights and Bias initialization
        # if self.load_weights == None:
        #     self.reset_parameters()
        # else:
        #     param, b = self.load_weights
        #     self.bias = torch.nn.Parameter(b)
        #     self.weight = torch.nn.Parameter(param, requires_grad=True)

        n, m = self.out_channels, self.in_channels * self.kernel_size_number
        # base = torch.randn(self.low_rank, requires_grad=False)
        # _, s_ordered, _ = torch.linalg.svd(torch.diag(torch.abs(base)))
        # U = torch.randn(n, self.low_rank, requires_grad=False)
        # V = torch.randn(m, self.low_rank, requires_grad=False)
        # U, _, _ = torch.linalg.svd(U)
        # V, _, _ = torch.linalg.svd(V)
        # self.s_hat = torch.nn.Parameter(torch.diag(s_ordered).to(device))

        base = torch.empty((n, m), requires_grad=False, **factory_kwargs)
        nn.init.kaiming_uniform_(base, a=math.sqrt(5))

        U, s_ord, vh = torch.linalg.svd(base, full_matrices=True)

        self.u = torch.nn.Parameter(U[:, : self.low_rank].to(device), requires_grad=False)
        self.v = torch.nn.Parameter(vh.T[:, : self.low_rank], requires_grad=False)

        self.s_hat = torch.nn.Parameter(s_ord[: self.low_rank, : self.low_rank], requires_grad=True)
        self.k = torch.nn.Parameter(torch.empty(n, self.low_rank, **factory_kwargs), requires_grad=True)
        self.l = torch.nn.Parameter(  # noqa: E741
            torch.empty(m, self.low_rank, **factory_kwargs),
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        n, m = self.out_channels, self.in_channels * self.kernel_size_number
        base = torch.empty((n, m), requires_grad=False, device=self.k.device, dtype=self.k.dtype)
        nn.init.kaiming_uniform_(base, a=math.sqrt(5))

        u, s_ord, vh = torch.linalg.svd(base, full_matrices=True)
        # Originally, v is set with the left vectors, not the right
        # will set reshape the paramaters
        self.u.set_(u[:, : self.low_rank])
        self.v.set_(vh.T[:, : self.low_rank])
        self.s_hat.set_(
            torch.diag(s_ord[: self.low_rank]).to(device=self.s_hat.device, dtype=self.s_hat.dtype),
        )  # trainable

        # todo: set up these with the same prepro as would be done during training
        nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))  # trainable
        nn.init.kaiming_uniform_(self.l, a=math.sqrt(5))  # trainable
        nn.init.kaiming_uniform_(self.n_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.m_hat, a=math.sqrt(5))

        # for testing
        # self.original_weight = Parameter(self.weight.reshape(self.original_shape))
        if self.bias is not None:
            weight = torch.empty(
                (self.out_channels, self.in_channels // self.groups, *self.kernel_size),
                device=self.k.device,
                dtype=self.k.dtype,
            )
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(base)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            del weight

    def forward(self, input):
        """
        forward phase for the convolutional layer. It has to contain the three different
        phases for the steps 'K','L' and 'S' in order to be optimizable using dlrt.

        """

        batch_size = input.shape[0]

        inp_unf = F.unfold(
            input,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        ).to(self.device)

        # out_h = int(np.floor(((input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (
        #                 self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1))
        # out_w = int(np.floor(((input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (
        #                   self.kernel_size[1] - 1) - 1) / self.stride[1]) + 1))

        out_h = (
            (input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
        ) + 1
        out_w = (
            (input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            / self.stride[1]
        ) + 1

        if self.train_case == "k":
            out_unf = torch.linalg.multi_dot([inp_unf.transpose(1, 2), self.v, self.k.T])
        elif self.train_case == "l":
            out_unf = torch.linalg.multi_dot([inp_unf.transpose(1, 2), self.l, self.u.T])
        elif self.train_case == "s":
            out_unf = torch.linalg.multi_dot(
                [inp_unf.transpose(1, 2), self.v, self.s_hat.T, self.u.T],
            )
        else:
            raise ValueError(f"Invalude step value: {self.step}")

        if self.bias is not None:
            out_unf.add_(self.bias)
        else:
            out_unf.transpose_(1, 2)
        return out_unf.view(batch_size, self.out_channels, out_h, out_w)

    def _k_preprocess(self):
        self.k.set_(self.u @ self.s_hat)

    def _k_postprocess(self):
        u_hat, _ = torch.linalg.qr(self.k)
        self.m_hat.set_(u_hat.T @ self.u)
        self.u.set_(u_hat)

    def _l_preprocess(self):
        self.l.set_(self.v @ self.s_hat.T)

    def _l_postprocess(self):
        v_hat, _ = torch.linalg.qr(self.l)
        self.n_hat.set_(v_hat.T @ self.v)
        self.v.data = v_hat

    def _s_preprocess(self):
        self.s_hat.set_(torch.linalg.multi_dot([self.m_hat, self.s_hat, self.n_hat.T]))


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
        eps_adapt: float = 0.1,
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
        )
        self.train_case = "k"
        self.eps_adapt = eps_adapt
        if self.bias is not None:
            self.bias = torch.nn.Parameter(
                torch.zeros(self.out_channels, requires_grad=False, **factory_kwargs),
                requires_grad=False,
            )
        # Weights and Bias initialization
        # if self.load_weights == None:
        #     self.reset_parameters()
        # else:
        #     param, b = self.load_weights
        #     self.bias = torch.nn.Parameter(b)
        #     self.weight = torch.nn.Parameter(param, requires_grad=True)

        n, m = self.out_channels, self.in_channels * self.kernel_size_number
        # base = torch.randn(self.low_rank, requires_grad=False)
        # _, s_ordered, _ = torch.linalg.svd(torch.diag(torch.abs(base)))
        # U = torch.randn(n, self.low_rank, requires_grad=False)
        # V = torch.randn(m, self.low_rank, requires_grad=False)
        # U, _, _ = torch.linalg.svd(U)
        # V, _, _ = torch.linalg.svd(V)
        # self.s_hat = torch.nn.Parameter(torch.diag(s_ordered).to(device))

        base = torch.empty((n, m), requires_grad=False, **factory_kwargs)
        nn.init.kaiming_uniform_(base, a=math.sqrt(5))

        U, s_ord, vh = torch.linalg.svd(base, full_matrices=True)

        self.u = torch.nn.Parameter(U[:, : self.low_rank].to(device), requires_grad=False)
        self.v = torch.nn.Parameter(vh.T[:, : self.low_rank], requires_grad=False)

        self.s_hat = torch.nn.Parameter(
            torch.diag(s_ord[: self.low_rank]).to(**factory_kwargs),
            requires_grad=True,
        )
        self.k = torch.nn.Parameter(torch.empty(n, self.low_rank, **factory_kwargs), requires_grad=True)
        self.l = torch.nn.Parameter(  # noqa: E741
            torch.empty(m, self.low_rank, **factory_kwargs),
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

        self.u_hat = torch.nn.Parameter(
            torch.empty(n, 2 * self.rmax).to(device),
            requires_grad=False,
        )
        self.v_hat = torch.nn.Parameter(
            torch.empty(m, 2 * self.rmax).to(device),
            requires_grad=False,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        n, m = self.out_channels, self.in_channels * self.kernel_size_number
        base = torch.empty(
            (n, m),
            requires_grad=False,
            device=self.k.data.device,
            dtype=self.k.data.dtype,
        )
        nn.init.kaiming_uniform_(base, a=math.sqrt(5))

        u, s_ord, vh = torch.linalg.svd(base, full_matrices=True)
        # Originally, v is set with the left vectors, not the right
        # will set reshape the paramaters
        self.u.set_(u[:, : self.low_rank])
        self.v.set_(vh.T[:, : self.low_rank])
        self.s_hat.set_(
            torch.diag(s_ord[: self.low_rank]).to(device=self.s_hat.device, dtype=self.s_hat.dtype),
        )  # trainable

        nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))  # trainable
        nn.init.kaiming_uniform_(self.l, a=math.sqrt(5))  # trainable
        nn.init.kaiming_uniform_(self.n_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.m_hat, a=math.sqrt(5))

        # todo: fix these to be like prepro steps!
        nn.init.kaiming_uniform_(self.u_hat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_hat, a=math.sqrt(5))

        # for testing
        # self.original_weight = Parameter(self.weight.reshape(self.original_shape))
        if self.bias is not None:
            weight = torch.empty(
                (self.out_channels, self.in_channels // self.groups, *self.kernel_size),
                device=self.k.device,
                dtype=self.k.dtype,
            )
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(base)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            del weight

    def forward(self, input):
        """
        forward phase for the convolutional layer. It has to contain the three different
        phases for the steps 'K','L' and 'S' in order to be optimizable using dlrt.

        """

        batch_size = input.shape[0]

        inp_unf = F.unfold(
            input,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        ).to(input.device)

        # out_h = int(np.floor(((input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (
        #                 self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1))
        # out_w = int(np.floor(((input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (
        #                   self.kernel_size[1] - 1) - 1) / self.stride[1]) + 1))

        out_h = (
            int(
                (input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                // self.stride[0],
            )
            + 1
        )
        out_w = (
            int(
                (input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                / self.stride[1],
            )
            + 1
        )

        if self.train_case == "k":
            k, v = self.k[:, : self.low_rank], self.v[:, : self.low_rank]
            # print([inp_unf.transpose(1, 2).shape, v.shape, k.T.shape], self.low_rank)
            # out_unf = torch.linalg.multi_dot([inp_unf.transpose(1, 2), v, k.T])
            out_unf = inp_unf.transpose(1, 2) @ v @ k.T
        elif self.train_case == "l":
            # out_unf = torch.linalg.multi_dot(
            #     [inp_unf.transpose(1, 2), self.l[:, : self.low_rank], self.u[:, : self.low_rank].T],
            # )
            out_unf = (
                inp_unf.transpose(1, 2)
                @ self.l[:, : self.low_rank]
                @ self.u[
                    :,
                    : self.low_rank,
                ].T
            )
        elif self.train_case == "s":
            u_hat = self.u_hat[:, : 2 * self.low_rank]
            s_hat = self.s_hat[: 2 * self.low_rank, : 2 * self.low_rank]
            v_hat = self.v_hat[:, : 2 * self.low_rank]
            # out_unf = torch.linalg.multi_dot(
            #     [inp_unf.transpose(1, 2), v_hat, s_hat.T, u_hat.T],
            # )
            out_unf = inp_unf.transpose(1, 2) @ v_hat @ s_hat.T @ u_hat.T
        else:
            raise ValueError(f"Invalude step value: {self.step}")

        if self.bias is not None:
            out_unf.add_(self.bias)
        else:
            out_unf.transpose_(1, 2)
        return out_unf.view(batch_size, self.out_channels, out_h, out_w)

    def k_preprocess(self):
        self.k.requires_grad = False
        self.l.requires_grad = False
        self.s_hat.requires_grad = False
        rnk = self.low_rank
        # K = self.u[:, :rnk] @ self.s_hat[:rnk, :rnk]
        # self.k[:, :rnk] = K
        # self.k.set_(self.u[:, :rnk] @ self.s_hat[:rnk, :rnk])
        self.k = nn.Parameter(self.u[:, :rnk] @ self.s_hat[:rnk, :rnk], requires_grad=True)
        # self.k.set_(self.u @ self.s_hat)

    def k_postprocess(self):
        self.k.requires_grad = False
        self.l.requires_grad = False
        self.s_hat.requires_grad = False

        # u_hat, _ = torch.linalg.qr(self.k)
        # self.m_hat.set_(u_hat.T @ self.u)
        # self.u.set_(u_hat)
        rnk = self.low_rank
        U_hat = torch.linalg.qr(torch.hstack((self.k[:, :rnk], self.u[:, :rnk])))[0]
        # self.u_hat[:, :2 * rnk] = U_hat
        # self.m_hat[:2 * rnk, :rnk] = self.u_hat[:, :2 * rnk].T @ self.u[:, :rnk]
        # self.u_hat.set_(U_hat)
        self.u_hat = nn.Parameter(U_hat, requires_grad=False)
        # self.m_hat.set_(self.u_hat.T @ self.u[:, :rnk])
        self.m_hat = nn.Parameter(self.u_hat.T @ self.u[:, :rnk], requires_grad=False)

    def l_preprocess(self):
        self.k.requires_grad = False
        self.l.requires_grad = False
        self.s_hat.requires_grad = False
        rnk = self.low_rank
        # L = self.v[:, :rnk] @ self.s_hat[:rnk, :rnk].T
        # self.l[:, :rnk] = L
        # self.l.set_(self.v[:, :rnk] @ self.s_hat[: rnk, : rnk].T)
        self.l = nn.Parameter(self.v[:, :rnk] @ self.s_hat[:rnk, :rnk].T, requires_grad=True)  # noqa: E741
        # self.l.set_(self.v @ self.s_hat.T)

    def l_postprocess(self):
        self.k.requires_grad = False
        self.l.requires_grad = False
        self.s_hat.requires_grad = False

        rnk = self.low_rank
        # v_hat, _ = torch.linalg.qr(self.l)
        # self.n_hat.set_(v_hat.T @ self.v)
        # self.v.data = v_hat
        v_hat = torch.linalg.qr(torch.hstack((self.l[:, :rnk], self.v[:, :rnk])))[0]
        # self.v_hat[:, :2 * rnk] = v_hat
        # self.n_hat[:2 * rnk, :rnk] = self.v_hat[:, :2 * rnk].T @ self.v[:, :rnk]
        # self.v_hat.set_(v_hat)
        self.v_hat = nn.Parameter(v_hat, requires_grad=False)
        # self.n_hat.set_(self.v_hat.T @ self.v)
        self.n_hat = nn.Parameter(self.v_hat.T @ self.v, requires_grad=False)

    def s_preprocess(self):
        self.k.requires_grad = False
        self.l.requires_grad = False
        self.s_hat.requires_grad = False

        rnk = self.low_rank
        # self.s_hat.set_(torch.linalg.multi_dot([self.m_hat, self.s_hat, self.n_hat.T]))
        s = self.m_hat[: 2 * rnk, :rnk] @ self.s_hat[:rnk, :rnk] @ self.n_hat[: 2 * rnk, :rnk].T
        # self.s_hat[:2 * rnk, :2 * rnk] = s
        # self.s_hat.set_(s)
        self.s_hat = nn.Parameter(s, requires_grad=True)

    @torch.no_grad()
    def rank_adaption(self):
        # print("running rank_adaption")
        rnk = self.low_rank
        s_small = self.s_hat[: 2 * rnk, : 2 * rnk]
        u2, d, v2 = torch.linalg.svd(s_small)

        # tol = self.theta * torch.linalg.norm(d)  # if not self.absolute else self.theta
        tol = self.eps_adapt * torch.linalg.norm(d)
        rmax = int(np.floor(d.shape[0] / 2))
        for j in range(0, 2 * rmax - 1):
            tmp = torch.linalg.norm(d[j : 2 * rmax - 1])
            if tmp < tol:
                rmax = j
                break

        rmax = min([rmax, self.rmax])
        rmax = max([rmax, 2])

        self.s_hat = nn.Parameter(torch.diag(d[:rmax]).to(device=self.s_hat.device, dtype=self.s_hat.dtype))
        # self.u[:, :rmax] = l.U_hat[:, :2 * rnk] @ u2[:, :rmax]
        self.u = nn.Parameter(self.u_hat[:, : 2 * rnk] @ u2[:, :rmax], requires_grad=False)
        # self.v[:, :rmax] = l.V_hat[:, :2 * rnk] @ (v2[:, :rmax])
        self.v = nn.Parameter(self.v_hat[:, : 2 * rnk] @ (v2[:, :rmax]), requires_grad=False)
        self.low_rank = int(rmax)
