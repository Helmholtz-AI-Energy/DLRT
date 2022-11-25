from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

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
        lowrank = int(low_rank_percent * in_features * out_features) if low_rank_percent is not None else None
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
            torch.empty((self.in_features, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.unp1 = nn.Parameter(
            torch.empty((self.in_features, 2 * self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.vt = nn.Parameter(
            torch.empty((self.rmax, self.out_features), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.vtnp1 = nn.Parameter(
            torch.empty((2 * self.rmax, self.out_features), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.n = nn.Parameter(
            torch.empty((2 * self.rmax, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.m = nn.Parameter(
            torch.empty((2 * self.rmax, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )

        self.reset_parameters()
        self.train_case = "k"

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, rank={self.low_rank}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )

    def get_classic_weight_repr(self):
        return self.k @ self.s @ self.lt

    @torch.no_grad()
    def set_aux_vars(self):
        factory = {"device": self.k.device, "dtype": self.k.dtype}
        # u, vt, n, m, unp1, vtprev

        self.u = nn.Parameter(torch.linalg.qr(self.k)[0].to(**factory), requires_grad=False)
        # overwrite the old vars once there are new ones
        nn.init.kaiming_uniform_(self.unp1, a=math.sqrt(5))
        # aux_N -> aux_Unp1.T @ aux_U
        #   used in setting s,
        self.n = nn.Parameter(self.u.T @ self.unp1, requires_grad=False)
        # aux_Vtnp1 -> q from qr(lt.T)
        self.vt = nn.Parameter(torch.linalg.qr(self.lt.T)[0], requires_grad=False)
        nn.init.kaiming_uniform_(self.vtnp1, a=math.sqrt(5))
        # aux_M -> aux_Vtnp1 @ aux_Vt.T
        self.m = nn.Parameter(self.vt @ self.vtnp1.T, requires_grad=False)

        self.u.requires_grad = False
        self.vt.requires_grad = False
        self.n.requires_grad = False
        self.m.requires_grad = False
        self.unp1.requires_grad = False
        self.vtnp1.requires_grad = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.init_method == "random":
            nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.s, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lt, a=math.sqrt(5))
            if self.bias is not None:
                w = self.k @ self.s @ self.lt  # should it be lt.T??
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        else:  # svd init case
            weights = torch.empty(
                (self.in_features, self.out_features),
                device=self.k.device,
                dtype=self.k.dtype,
            )
            nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
            k, vec_s, ltt = torch.linalg.svd(weights)
            self.k *= 0
            self.k += k[:, : self.low_rank].clone().detach()
            self.s *= 0
            self.s += torch.diag(vec_s[: self.low_rank].clone().detach())
            self.lt *= 0
            # TODO: transpose needed??
            self.lt += ltt.T[:, : self.low_rank].clone().detach()
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        self.set_aux_vars()

    def forward(self, input: Tensor) -> Tensor:
        if self.train_case == "k" or not self.training:  # k-step
            ret = torch.linalg.multi_dot([input, self.k, self.vt])
        elif self.train_case == "l":  # l-step
            ret = torch.linalg.multi_dot([input, self.u, self.lt])
            # ret = (input @ self.u) @ self.lt
        else:  # s-step
            ret = torch.linalg.multi_dot([input, self.unp1, self.s, self.vtnp1])
            # ret = ((input @ self.unp1) @ self.s) @ self.vtnp1
        return ret if self.bias is None else ret + self.bias

    @torch.no_grad()
    def k_preprocess(self):
        self.lt.requires_grad = False
        self.s.requires_grad = False
        self.bias.requires_grad = False
        # k prepro
        # k -> aux_U @ s
        self.k = nn.Parameter(self.u @ self.s, requires_grad=True)

    @torch.no_grad()
    def l_preprocess(self):
        self.k.requires_grad = False
        self.s.requires_grad = False
        self.bias.requires_grad = False
        # lt -> s @ aux_Vt
        self.lt = nn.Parameter(self.s @ self.vt, requires_grad=True)

    @torch.no_grad()
    def k_postprocess(self):
        self.k.requires_grad = False
        self.s.requires_grad = False
        # aux_Unp1 -> q from qr(k)
        #   aux_Unp1 used in s-step forward, can keep in u
        self.u = nn.Parameter(torch.linalg.qr(self.k)[0], requires_grad=False)
        self.u.requires_grad = False
        # aux_N -> aux_Unp1.T @ aux_U
        #   used in setting s,
        self.n = nn.Parameter(self.u.T @ self.unp1, requires_grad=False)

    @torch.no_grad()
    def l_postprocess(self):
        self.lt.requires_grad = False
        self.s.requires_grad = False
        # aux_Vtnp1 -> q from qr(lt.T)
        self.vt = nn.Parameter(torch.linalg.qr(self.lt.T)[0], requires_grad=False)
        # aux_M -> aux_Vtnp1 @ aux_Vt.T
        self.m = nn.Parameter(self.vt @ self.vtnp1.T, requires_grad=False)
        # overwrite the old vars once there is are new ones
        self.unp1 = nn.Parameter(self.u.data, requires_grad=False)
        self.vtnp1 = nn.Parameter(self.vt.data, requires_grad=False)

    @torch.no_grad()
    def s_preprocess(self):
        self.k.requires_grad = False
        self.lt.requires_grad = False
        self.bias.requires_grad = True
        # set aux_U -> aux_Unp1  # done above
        # set aux_Vt -> aux_Vtnp1  # done previously now
        # set s -> (aux_N @ s) @ aux_M.T
        self.s = nn.Parameter(self.n @ self.s @ self.m.T, requires_grad=True)


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
        self.s = nn.Parameter(torch.empty((self.rmax, self.rmax), **factory_kwargs))
        self.lt = nn.Parameter(torch.empty((self.rmax, out_features), **factory_kwargs))

        self.u = nn.Parameter(
            torch.empty((self.in_features, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.unp1 = nn.Parameter(
            torch.empty((self.in_features, 2 * self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.vt = nn.Parameter(
            torch.empty((self.rmax, self.out_features), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.vtnp1 = nn.Parameter(
            torch.empty((2 * self.rmax, self.out_features), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.n = nn.Parameter(
            torch.empty((2 * self.rmax, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )
        self.m = nn.Parameter(
            torch.empty((2 * self.rmax, self.rmax), requires_grad=False, **factory_kwargs),
            requires_grad=False,
        )

        self.reset_parameters()
        self.train_case = "k"

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, low_rank={self.low_rank}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )

    @torch.no_grad()
    def set_aux_vars(self):
        factory = {"device": self.k.device, "dtype": self.k.dtype}
        lr = self.low_rank
        lr2 = 2 * lr
        # needed at top: u, vt
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.unp1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.m, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.vtnp1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.n, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.m, a=math.sqrt(5))
        # need to check sizes...
        ## from k postpro ---------------------------------------------------
        # k_extended = torch.cat((self.k[:, :lr], self.u[:, :lr]), dim=1)
        # prev_u, _ = torch.linalg.qr(k_extended)
        ## aux_N -> aux_Unp1.T @ aux_U
        ##self.unp1[:, :lr2] = prev_u
        ##self.unp1.requires_grad = False
        # self.u.requires_grad = False
        ##   used in setting s,
        # self.n[: 2 * lr, :lr] = self.unp1[:, :lr2].T @ self.u[:, :lr]
        # self.n.requires_grad = False
        ## from l postpro ---------------------------------------------------
        ## print(self.lt[:lr].T.shape, self.vt[:lr].shape)
        # l_extended = torch.cat((self.lt[:lr].T, self.vt[:lr].T), dim=1)
        ## l_extended = torch.cat((self.lt[:lr], self.vt[:lr]), dim=1)
        # aux_Vnp1, _ = torch.linalg.qr(l_extended)
        # self.vtnp1[:lr2] = aux_Vnp1.T
        # self.m[:lr2, :lr] = self.vtnp1[:lr2, :] @ self.vt[:lr].T
        # self.vtnp1.requires_grad = False
        # self.m.requires_grad = False
        #
        ## from rank adjustment ---------------------------------------------------
        # s_small = self.s[: 2 * self.low_rank, : 2 * self.low_rank]
        # u2, sing, vh2 = torch.linalg.svd(s_small, full_matrices=False)
        # v2 = vh2.T
        #
        # rmax = self.rmax
        ## update s
        # self.s.set_(torch.diag(sing[:rmax]).to(**factory))
        #
        ## create and update u and v
        # self.u.set_(self.unp1[:, : 2 * self.low_rank] @ u2[:, :rmax])
        # self.vt.set_(v2[:rmax, :] @ self.vtnp1[: 2 * self.low_rank, :])
        ## self.low_rank = int(rmax)

    def get_classic_weight_repr(self):
        return self.k @ self.s @ self.lt

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.init_method == "random":
            nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.s, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lt, a=math.sqrt(5))
            if self.bias is not None:
                w = self.k @ self.s @ self.lt  # should it be lt.T??
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        else:  # svd init case
            weights = torch.empty(
                (self.in_features, self.out_features),
                device=self.k.device,
                dtype=self.k.dtype,
            )
            nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
            k, vec_s, ltt = torch.linalg.svd(weights)
            self.k.set_(k[:, : self.low_rank].clone().detach())
            self.s.set_(torch.diag(vec_s[: self.low_rank].clone().detach()))
            self.lt.set_(ltt.T[:, : self.low_rank].clone().detach())
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        self.set_aux_vars()

    def change_training_case(self, case):
        # switch -> if current train case is k/l, do post for
        self.train_case = case

    def cycle_training_case(self):
        # cycle between the cases -> to be used in the trainer/model class
        if self.train_case == "k" or not self.training:
            self.train_case = "l"
        elif self.train_case == "l":
            self.kl_postpro_s_prepro()
            self.train_case = "s"
        else:
            self.kl_prepro()
            self.train_case = "k"

    # @torch.jit.script
    def forward(self, input: Tensor) -> Tensor:
        eps = torch.finfo(input.dtype).eps
        if self.train_case == "k":  # k-step
            # remove elements close to 0
            # second = self.k[:, : self.low_rank] @ self.vt[: self.low_rank]
            # second[(second >= eps) & (second <= -eps)] *= 0
            # ret = input @ second
            ret = torch.linalg.multi_dot(
                [input, self.k[:, : self.low_rank], self.vt[: self.low_rank]],
            )
            # ret = (input @ self.k[:, : self.low_rank]) @ self.vt[: self.low_rank]
        elif self.train_case == "l":  # l-step
            # second = self.u[:, : self.low_rank] @ self.lt[: self.low_rank]
            # second[(second >= eps) & (second <= -eps)] *= 0
            # ret = input @ second
            ret = torch.linalg.multi_dot(
                [input, self.u[:, : self.low_rank], self.lt[: self.low_rank]],
            )
            # ret = (input @ self.u[:, : self.low_rank]) @ self.lt[: self.low_rank]
        else:  # s-step
            lr2 = 2 * self.low_rank
            # second = torch.linalg.multi_dot(
            #    [self.unp1[:, :lr2], self.s[:lr2, :lr2], self.vtnp1[:lr2]],
            # )
            # second[(second >= eps) & (second <= -eps)] *= 0
            # ret = input @ second
            ret = torch.linalg.multi_dot(
                [input, self.unp1[:, :lr2], self.s[:lr2, :lr2], self.vtnp1[:lr2]],
            )
            # ret = ((input @ self.unp1[:, :lr2]) @ self.s[:lr2, :lr2]) @ self.vtnp1[:lr2]

        return ret if self.bias is None else ret + self.bias

    @torch.no_grad()
    def k_preprocess(self):
        self.k.requires_grad = False
        self.lt.requires_grad = False
        self.s.requires_grad = False

        self.bias.requires_grad = False
        # k prepro
        # k -> aux_U @ s
        kls = self.s[: self.low_rank, : self.low_rank]
        # self.k[:, : self.low_rank] = self.u[:, : self.low_rank] @ kls
        # self.k.set_(self.u[:, : self.low_rank] @ kls)
        self.k = nn.Parameter(self.u[:, : self.low_rank] @ kls, requires_grad=True)

    @torch.no_grad()
    def l_preprocess(self):
        self.k.requires_grad = False
        self.lt.requires_grad = False
        self.s.requires_grad = False
        # lt -> s @ aux_Vt
        self.bias.requires_grad = False
        kls = self.s[: self.low_rank, : self.low_rank]
        # self.lt[: self.low_rank] = kls @ self.vt[: self.low_rank]
        # self.lt.set_(kls @ self.vt[: self.low_rank])
        self.lt = nn.Parameter(kls @ self.vt[: self.low_rank], requires_grad=True)

    @torch.no_grad()
    def k_postprocess(self):
        self.k.requires_grad = False
        self.lt.requires_grad = False
        self.s.requires_grad = False
        lr = self.low_rank
        lr2 = 2 * self.low_rank
        # ------- k postpro ----------------------------------
        # aux_Unp1 -> q from qr(k)
        #   aux_Unp1 used in s-step forward, can keep in u
        k_extended = torch.cat((self.k[:, :lr], self.u[:, :lr]), dim=1)
        prev_u, _ = torch.linalg.qr(k_extended)
        # aux_N -> aux_Unp1.T @ aux_U
        self.unp1[:, :lr2] = prev_u
        # NOTE: unp1 -> [512, 4608] - prev_u = [512, 512]
        self.unp1.requires_grad = False
        self.u.requires_grad = False
        #   used in setting s,
        # TODO: check me! unp1 might be wrong here...
        aux_n = self.unp1[:, :lr2].T @ self.u[:, :lr]
        self.n[: 2 * lr, :lr] = aux_n
        self.n.requires_grad = False

    @torch.no_grad()
    def l_postprocess(self):
        self.k.requires_grad = False
        self.lt.requires_grad = False
        self.s.requires_grad = False
        lr = self.low_rank
        lr2 = 2 * self.low_rank
        l_extended = torch.cat((self.lt[:lr].T, self.vt[:lr].T), dim=1)
        aux_Vnp1, _ = torch.linalg.qr(l_extended)
        self.vtnp1[:lr2] = aux_Vnp1.T
        m = self.vtnp1[:lr2, :] @ self.vt[:lr].T
        self.m[:lr2, :lr] = m
        self.vtnp1.requires_grad = False
        self.m.requires_grad = False

    @torch.no_grad()
    def s_preprocess(self):
        self.k.requires_grad = False
        self.lt.requires_grad = False

        lr = self.low_rank
        lr2 = 2 * self.low_rank
        # bias is trainable for the s step
        self.bias.requires_grad = True
        # set s -> (aux_N @ s) @ aux_M.T
        self.s[:lr2, :lr2] = self.n[:lr2, :lr] @ self.s[:lr, :lr] @ self.m[:lr2, :lr].T
        self.s.requires_grad = True

    @torch.no_grad()
    def rank_adaption(self):
        # 1) compute SVD of S
        # d=singular values, u2 = left singuar vecs, v2= right singular vecs
        # TODO: 64 bit?
        s_small = self.s[: 2 * self.low_rank, : 2 * self.low_rank]
        try:
            u2, sing, vh2 = torch.linalg.svd(s_small.to(torch.float64), full_matrices=False, driver="gesvdj")
        except torch._C._LinAlgError as e:
            print(f"LinAlgError during SGD -> {e}")
            return
        v2 = vh2.T.to(self.s.dtype, non_blocking=True)
        u2 = u2.to(self.s.dtype, non_blocking=True)
        # d, u2, v2 = tf.linalg.svd(s_small)

        # absolute value treshold (try also relative one)
        tol = self.eps_adapt * torch.linalg.norm(sing)
        rmax = sing.shape[0] // 2
        for j in range(2, 2 * rmax - 1):
            tmp = torch.linalg.norm(sing[j : 2 * rmax - 1])
            if tmp < tol:
                rmax = j
                break

        # rmax = max(min(rmax, self.rmax), 2)  # -> handled by the 2 in the for loop above
        new_sz = int(rmax * 2)
        # update s
        # self.s[:rmax, :rmax] = torch.diag(sing[:rmax]).to(device=self.s.device, dtype=self.s.dtype)
        # self.s.set_(torch.diag(sing[:rmax]).to(device=self.s.device, dtype=self.s.dtype))
        self.s = nn.Parameter(torch.diag(sing[:new_sz]).to(device=self.s.device, dtype=self.s.dtype))

        # update u and v
        # self.u[:, :rmax] = self.unp1[:, : 2 * self.low_rank] @ u2[:, :rmax]
        # self.vt[:rmax, :] = v2[:rmax, :] @ self.vtnp1[: 2 * self.low_rank, :]

        # self.u.set_(self.unp1[:, : 2 * self.low_rank] @ u2[:, :rmax])
        self.u = nn.Parameter(self.unp1[:, : 2 * self.low_rank] @ u2[:, :new_sz], requires_grad=False)
        # self.vt.set_(v2[:rmax, :] @ self.vtnp1[: 2 * self.low_rank, :])
        self.vt = nn.Parameter(v2[:new_sz, :] @ self.vtnp1[: 2 * self.low_rank, :], requires_grad=False)
        # self.vt = nn.Parameter(new_vt, requires_grad=False)
        self.low_rank = int(rmax)
