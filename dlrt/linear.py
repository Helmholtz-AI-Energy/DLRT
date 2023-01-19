from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.pretty import Pretty
from torch import Tensor

console = Console(width=140)

from .basic import DLRTModule

__all__ = ["DLRTLinear", "DLRTLinearFixed", "DLRTLinearAdaptive"]


def DLRTLinear(
    in_features: int,
    out_features: int,
    adaptive: bool = True,
    low_rank_percent: int = None,
    bias: bool = True,
    init_method: str = "random",
    device=None,
    dtype=None,
    eps_adapt: float = 0.01,
):
    """
    Gets a linear layer with the given features

    Args:
    """
    if not adaptive:
        if low_rank_percent is not None:
            lowrank = int(low_rank_percent * min(in_features, out_features))
        else:
            lowrank = None
        return DLRTLinearFixed(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            low_rank=lowrank,
            init_method=init_method,
            device=device,
            dtype=dtype,
        )
    else:
        # return DLRTLinearAdaptiveTransposed(
        return DLRTLinearAdaptive(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            low_rank_percent=low_rank_percent,
            eps_adapt=eps_adapt,
            device=device,
            dtype=dtype,
        )


class DLRTLinearFixed(DLRTModule):
    # should this instead inherit from nn.Linear?
    #   doesnt need to, everything in nn.Linear is overwritten
    # overwrite the original layer depending on its type?
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        low_rank: int = None,
        bias: bool = True,
        init_method: str = "random",
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
        low_rank:
            top-rank approx. this will cut to the top-rank eigenvectors
            for the math, this is the inner dim of the decomp
        """

        self.low_rank = low_rank if low_rank is not None else min([in_features, out_features])
        if low_rank > in_features:
            raise ValueError(
                f"rank > in_features ({low_rank} > {in_features}) use nn.Linear or reduce rank",
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if (isinstance(bias, bool) and bias) or bias is not None:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.basic_number_weights = out_features * in_features

        self.dlrt = True
        self.init_method = init_method
        assert init_method in ["random", "svd"], "init_method must be in ['random', 'svd']"

        # need k, lt, s, bias
        # K -> U @ S, L -> V @ S.T
        self.k = nn.Parameter(torch.empty((in_features, low_rank), **factory_kwargs))
        self.s = nn.Parameter(torch.empty((low_rank, low_rank), **factory_kwargs))
        self.lt = nn.Parameter(torch.empty((low_rank, out_features), **factory_kwargs))

        self.u = nn.Parameter(
            torch.empty((self.in_features, self.low_rank), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.unp1 = nn.Parameter(
            torch.empty((self.in_features, self.low_rank), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.vt = nn.Parameter(
            torch.empty((self.low_rank, self.out_features), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.vtnp1 = nn.Parameter(
            torch.empty((self.low_rank, self.out_features), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.n = nn.Parameter(
            torch.empty((self.low_rank, self.low_rank), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.m = nn.Parameter(
            torch.empty((self.low_rank, self.low_rank), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )

        self.reset_parameters()
        self.train_case = "k"

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, rank={self.low_rank}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )

    def print_means(self):
        shapes = []
        shapes.append(f"k: {self.k.mean():.4f} {self.k.min():.4f} {self.k.max():.4f} {self.k.requires_grad}")
        shapes.append(
            f"s: {self.s.mean():.4f} {self.s.min():.4f} {self.s.max():.4f} " f"{self.s.requires_grad}",
        )
        shapes.append(
            f"lt: {self.lt.mean():.4f} {self.lt.min():.4f} {self.lt.max():.4f}" f" {self.lt.requires_grad}",
        )
        shapes.append(
            f"u: {self.u.mean():.4f} {self.u.min():.4f} {self.u.max():.4f} " f"{self.u.requires_grad}",
        )
        shapes.append(
            f"unp1: {self.unp1.mean():.4f} {self.unp1.min():.4f} "
            f"{self.unp1.max():.4f} {self.unp1.requires_grad}",
        )
        shapes.append(
            f"vt: {self.vt.mean():.4f} {self.vt.min():.4f} {self.vt.max():.4f}" f" {self.vt.requires_grad}",
        )
        shapes.append(
            f"vtnp1: {self.vtnp1.mean():.4f} {self.vtnp1.min():.4f} "
            f"{self.vtnp1.max():.4f} {self.vtnp1.requires_grad}",
        )
        shapes.append(
            f"n: {self.n.mean():.4f} {self.n.min():.4f} " f"{self.n.max():.4f} {self.n.requires_grad}",
        )
        shapes.append(
            f"m: {self.m.mean():.4f} {self.m.min():.4f} " f"{self.m.max():.4f} {self.m.requires_grad}",
        )
        if self.bias is not None:
            shapes.append(
                f"bias: {self.bias.mean():.4f} {self.bias.min():.4f} "
                f"{self.bias.max():.4f} {self.bias.requires_grad}",
            )
        # if self.rank == 0: # and self.counter % 100 == 0:
        columns = Columns(shapes, equal=True, expand=True)
        # console.rule("All shapes in linear")
        console.print(columns)

    def get_classic_weight_repr(self):
        if self.s.ndim == 1:
            return self.k @ torch.diag(self.s) @ self.lt
        return self.k @ self.s @ self.lt

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.s, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.vt, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.unp1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.vtnp1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lt, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.n, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.m, a=math.sqrt(5))

        if self.bias is not None:
            w = torch.linalg.multi_dot([self.k, self.s, self.lt])
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # print('train case', self.train_case)
        # self.print_means()
        if self.train_case == "k" or not self.training:  # k-step
            ret = torch.linalg.multi_dot([input, self.k, self.vt])
        elif self.train_case == "l":  # l-step
            ret = torch.linalg.multi_dot([input, self.u, self.lt])
            # ret = (input @ self.u) @ self.lt
        else:  # s-step
            ret = torch.linalg.multi_dot([input, self.unp1, self.s, self.vtnp1])
            # ret = ((input @ self.unp1) @ self.s) @ self.vtnp1
        return ret if self.bias is None else ret + self.bias

    def _change_params_requires_grad(self, requires_grad):
        self.u.requires_grad = False  # requires_grad
        self.s.requires_grad = requires_grad
        self.vt.requires_grad = False  # requires_grad
        self.unp1.requires_grad = False  # requires_grad
        self.vtnp1.requires_grad = False  # requires_grad
        self.k.requires_grad = requires_grad
        self.lt.requires_grad = requires_grad
        self.n.requires_grad = False  # requires_grad
        self.m.requires_grad = False  # requires_grad
        self.bias.requires_grad = requires_grad

    @torch.no_grad()
    def k_preprocess(self):
        self._change_params_requires_grad(False)
        # k prepro
        # k -> aux_U @ s
        # self.k = nn.Parameter(self.u @ self.s, requires_grad=True)
        self.k.set_(self.u @ self.s)
        # TODO: !! sorting s at the top of the training loop might fuck everything up
        # self.s.set_(torch.sort(self.s, descending=True).values)

        self.k.requires_grad = True

    @torch.no_grad()
    def l_preprocess(self):
        self._change_params_requires_grad(False)
        # lt -> s @ aux_Vt
        self.lt.set_(self.s @ self.vt)
        self.lt.requires_grad = True

    @torch.no_grad()
    def k_postprocess(self):
        # NOTE must run after 'l' forward step b/c self.u is used in the l forward step
        self._change_params_requires_grad(False)
        # aux_Unp1 -> q from qr(k)
        #   aux_Unp1 used in s-step forward, can keep in u
        self.unp1.set_(torch.linalg.qr(self.k)[0])
        # aux_N -> aux_Unp1.T @ aux_U
        #   used in setting s,
        self.n.set_(self.unp1.T @ self.u)

    @torch.no_grad()
    def l_postprocess(self):
        self._change_params_requires_grad(False)
        # aux_Vtnp1 -> q from qr(lt.T)
        self.vtnp1.set_(torch.linalg.qr(self.lt.T)[0].T)
        # aux_M -> aux_Vtnp1 @ aux_Vt.T
        self.m.set_(self.vtnp1 @ self.vt.T)

    @torch.no_grad()
    def s_preprocess(self):
        self._change_params_requires_grad(False)
        if self.bias is not None:
            self.bias.requires_grad = True
        # set aux_U -> aux_Unp1  # done above
        # set aux_Vt -> aux_Vtnp1  # done previously now
        # set s -> (aux_N @ s) @ aux_M.T
        # self.s = nn.Parameter(self.n @ self.s @ self.m.T, requires_grad=True)
        self.s.set_(self.n @ self.s @ self.m.T)
        self.s.requires_grad = True
        # overwrite the old vars once there is are new ones
        self.u.set_(self.unp1.data)
        self.vt.set_(self.vtnp1.data)


class DLRTLinearAdaptive(DLRTModule):
    # overwrite the original layer depending on its type?
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        low_rank_percent: float = None,
        # rmax: int = None,
        bias: bool = True,
        eps_adapt: float = 0.1,
        init_method: str = "random",
        device=None,
        dtype=None,
    ) -> None:
        """

        Parameters
        ----------
        in_features
        out_features
        low_rank_percent
            starting inner rank
        rmax
            max number of ranks (percentage)
        bias
        eps_adapt
            epsilon to use in adaptive methods.
        init_method
            init with svd or with random for K, L.T, and S
        device
        dtype
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        if (isinstance(bias, bool) and bias) or bias is not None:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if low_rank_percent is None:
            # set the max low_rank to be such that the
            roots = np.roots([1, in_features + out_features, in_features * out_features])
            pos_coeff = roots[roots > 0]  # TODO: adjust factor?
            if len(pos_coeff) < 1:
                self.rmax = min([in_features, out_features]) // 2
            else:
                self.rmax = int(np.floor(pos_coeff[-1]))
            # set the initial low_rank to be most of the rmax
            if self.rmax < 10:
                self.rmax = 20
            self.low_rank = self.rmax // 2
        else:
            self.rmax = min([in_features, out_features]) // 2
            self.low_rank = int(self.rmax * low_rank_percent)
            self.rmax = int(self.low_rank * 2)  # TODO: cleanup?
            print(self.rmax, self.low_rank)

        self.basic_number_weights = out_features * in_features

        self.eps_adapt = eps_adapt

        # int(low_rank_percent * min([in_features, out_features]))
        # self.rmax = min(rmax, int(min([in_features, out_features]) / 2))
        # self.low_rank = low_rank if low_rank is not None else min([in_features, out_features])

        self.dlrt = True
        self.init_method = init_method
        assert init_method in ["random", "svd"], "init_method must be in ['random', 'svd']"

        # need k, lt, s, bias
        # K -> U @ S, L -> V @ S.T
        self.k = nn.Parameter(torch.empty((in_features, self.rmax), **factory_kwargs))
        self.s = nn.Parameter(torch.empty((2 * self.rmax, 2 * self.rmax), **factory_kwargs))
        self.lt = nn.Parameter(torch.empty((self.rmax, out_features), **factory_kwargs))

        self.u = nn.Parameter(
            torch.zeros((self.in_features, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.unp1 = nn.Parameter(
            torch.zeros((self.in_features, 2 * self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.vt = nn.Parameter(
            torch.zeros((self.rmax, self.out_features), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.vtnp1 = nn.Parameter(
            torch.zeros((2 * self.rmax, self.out_features), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.n = nn.Parameter(
            torch.zeros((2 * self.rmax, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.m = nn.Parameter(
            torch.zeros((2 * self.rmax, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )

        self.reset_parameters()
        self.train_case = "k"

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, low_rank={self.low_rank}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )

    def get_classic_weight_repr(self):
        return self.k @ self.s @ self.lt

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.s, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.vt, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.unp1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.vtnp1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lt, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.n, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.m, a=math.sqrt(5))
        if self.bias is not None:
            w = torch.linalg.multi_dot([self.k, self.s[: self.rmax, : self.rmax], self.lt])
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _change_params_requires_grad(self, requires_grad):
        self.k.requires_grad = requires_grad
        self.s.requires_grad = requires_grad
        self.lt.requires_grad = requires_grad
        self.u.requires_grad = False  # requires_grad
        self.unp1.requires_grad = False  # requires_grad
        self.vt.requires_grad = False  # requires_grad
        self.vtnp1.requires_grad = False  # requires_grad
        self.n.requires_grad = False  # requires_grad
        self.m.requires_grad = False  # requires_grad
        self.bias.requires_grad = requires_grad

    def change_training_case(self, case):
        # switch -> if current train case is k/l, do post for
        self.train_case = case

    # @torch.jit.script
    def forward(self, input: Tensor) -> Tensor:
        # eps = torch.finfo(input.dtype).eps
        if self.train_case == "k":  # k-step
            # remove elements close to 0
            # second = self.k[:, : self.low_rank] @ self.vt[: self.low_rank]
            # second[(second >= eps) & (second <= -eps)] *= 0
            # ret = input @ second
            ret = torch.linalg.multi_dot(
                [input, self.k[:, : self.low_rank], self.vt[: self.low_rank]],
            )
        elif self.train_case == "l":  # l-step
            # second = self.u[:, : self.low_rank] @ self.lt[: self.low_rank]
            # second[(second >= eps) & (second <= -eps)] *= 0
            # ret = input @ second
            ret = torch.linalg.multi_dot(
                [input, self.u[:, : self.low_rank], self.lt[: self.low_rank]],
            )
        else:  # s-step
            # TODO: should this be only low_rank??? (not x2)
            lr2 = 2 * self.low_rank
            # second = torch.linalg.multi_dot(
            #    [self.unp1[:, :lr2], self.s[:lr2, :lr2], self.vtnp1[:lr2]],
            # )
            # second[(second >= eps) & (second <= -eps)] *= 0
            # ret = input @ second
            ret = torch.linalg.multi_dot(
                [input, self.unp1[:, :lr2], self.s[:lr2, :lr2], self.vtnp1[:lr2]],
            )

        return ret if self.bias is None else ret + self.bias

    @torch.no_grad()
    def k_preprocess(self):
        self._change_params_requires_grad(False)
        # k -> aux_U @ s
        s = self.s[: self.low_rank, : self.low_rank]

        # self.k.zero_()
        self.k[:, : self.low_rank] = self.u[:, : self.low_rank] @ s
        self.k.requires_grad = True
        self.k.training = True

    @torch.no_grad()
    def l_preprocess(self):
        self._change_params_requires_grad(False)
        s = self.s[: self.low_rank, : self.low_rank]
        # self.lt.zero_()
        self.lt[: self.low_rank] = s @ self.vt[: self.low_rank]
        self.lt.requires_grad = True
        self.lt.training = True

    @torch.no_grad()
    def k_postprocess(self):
        lr = self.low_rank
        lr2 = 2 * self.low_rank
        # ------- k postpro ----------------------------------
        # aux_Unp1 -> q from qr(k)
        #   aux_Unp1 used in s-step forward, can keep in u
        k_extended = torch.cat((self.k[:, :lr], self.u[:, :lr]), dim=1)
        prev_u, _ = torch.linalg.qr(k_extended)
        # aux_N -> aux_Unp1.T @ aux_U
        # self.unp1.zero_()
        self.unp1[:, :lr2] = prev_u
        #   used in setting s,
        # self.n.zero_()
        self.n[:lr2, :lr] = self.unp1[:, :lr2].T @ self.u[:, :lr]

    @torch.no_grad()
    def l_postprocess(self):
        lr = self.low_rank
        lr2 = 2 * self.low_rank
        l_extended = torch.cat((self.lt[:lr].T, self.vt[:lr].T), dim=1)
        aux_Vnp1, _ = torch.linalg.qr(l_extended)
        # self.vtnp1.zero_()
        self.vtnp1[:lr2] = aux_Vnp1.T
        self.m.zero_()
        self.m[:lr2, :lr] = self.vtnp1[:lr2] @ self.vt[:lr].T

    @torch.no_grad()
    def s_preprocess(self):
        self._change_params_requires_grad(False)
        lr = self.low_rank
        lr2 = 2 * self.low_rank
        self.s[:lr2, :lr2] = self.n[:lr2, :lr] @ self.s[:lr, :lr] @ self.m[:lr2, :lr].T
        # self.s.zero_()  # todo: ??
        self.s.requires_grad = True
        self.s.training = True
        self.bias.requires_grad = True
        self.bias.training = True

    @torch.no_grad()
    def rank_adaption(self, skip=False):
        # 1) compute SVD of S
        # d=singular values, u2 = left singuar vecs, v2= right singular vecs
        # TODO: 64 bit?
        s_small = self.s[: 2 * self.low_rank, : 2 * self.low_rank]
        try:
            u2, sing, vh2 = torch.linalg.svd(s_small, full_matrices=False)
            # driver="gesvdj")
        except torch._C._LinAlgError as e:
            print(f"LinAlgError during SGD -> {e}")
            return
        v2 = vh2.T.to(self.s.dtype, non_blocking=True)
        u2 = u2.to(self.s.dtype, non_blocking=True)
        # d, u2, v2 = tf.linalg.svd(s_small)

        if not skip:
            # absolute value treshold (try also relative one)
            tol = self.eps_adapt * torch.linalg.norm(sing)
            new_lr = sing.shape[0] // 2
            max_lr = 2 * new_lr - 1
            for j in range(2, max_lr):
                tmp = torch.linalg.norm(sing[j : 2 * new_lr - 1])
                if tmp < tol:
                    new_lr = j
                    break
        else:
            new_lr = self.low_rank

        # rmax = max(min(rmax, self.rmax), 2)  # -> handled by the 2 in the for loop above
        # new_sz = int(new_lr * 2)
        # update s
        # self.s.zero_()
        self.s[:new_lr, :new_lr] = torch.diag(sing[:new_lr]).to(device=self.s.device, dtype=self.s.dtype)
        # self.u.zero_()
        self.u[:, :new_lr] = self.unp1[:, : 2 * self.low_rank] @ u2[:, :new_lr]
        # self.vt.zero_()
        self.vt[:new_lr] = v2[:new_lr, :] @ self.vtnp1[: 2 * self.low_rank, :]
        self.low_rank = int(new_lr)
