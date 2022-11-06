from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from basic import DLRTModule


class DLRTLinear(DLRTModule):
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
        load_weights
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
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.dlrt = True
        self.init_method = init_method
        assert init_method in ["random", "svd"], "init_method must be in ['random', 'svd']"

        # need k, lt, s, bias
        # K -> U @ S, L -> V @ S.T
        self.k = nn.Parameter(torch.empty((in_features, low_rank), **factory_kwargs))
        self.s = nn.Parameter(torch.empty((low_rank, low_rank), **factory_kwargs))
        self.lt = nn.Parameter(torch.empty((low_rank, out_features), **factory_kwargs))

        self.reset_parameters()
        self.train_case = "k"

        self.set_aux_vars()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, rank={self.low_rank}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )

    def set_aux_vars(self):
        with torch.no_grad():
            factory_kwargs = {"device": self.k.device, "dtype": self.k.dtype}
            # u, vt, n, m, unp1, vtprev
            self.u, _ = torch.linalg.qr(self.k)
            self.u.requires_grad = False
            # overwrite the old vars once there are new ones
            self.unp1 = torch.empty(tuple(self.u.shape), requires_grad=False, **factory_kwargs)
            nn.init.kaiming_uniform_(self.unp1, a=math.sqrt(5))
            # aux_N -> aux_Unp1.T @ aux_U
            #   used in setting s,
            self.n = self.u.T @ self.unp1
            self.n.requires_grad = False
            # aux_Vtnp1 -> q from qr(lt.T)
            self.vt, _ = torch.linalg.qr(self.lt.T)
            self.vt.requires_grad = False
            self.vtnp1 = torch.empty(tuple(self.vt.shape), requires_grad=False, **factory_kwargs)
            nn.init.kaiming_uniform_(self.vtnp1, a=math.sqrt(5))
            # aux_M -> aux_Vtnp1 @ aux_Vt.T
            self.m = self.vt @ self.vtnp1.T

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
            ret = (input @ self.k) @ self.vt
        elif self.train_case == "l":  # l-step
            ret = (input @ self.u) @ self.lt
        else:  # s-step
            ret = ((input @ self.unp1) @ self.s) @ self.vtnp1
        return ret if self.bias is None else ret + self.bias

    @torch.no_grad()
    def kl_prepro(self):
        # disable bias training
        self.bias.requires_grad = False
        # k prepro
        # k -> aux_U @ s
        self.k = self.u @ self.s
        # l prepro
        # lt -> s @ aux_Vt
        self.lt = self.s @ self.vt

    @torch.no_grad()
    def kl_postpro_s_prepro(self):
        # ------- k postpro ----------------------------------
        # aux_Unp1 -> q from qr(k)
        #   aux_Unp1 used in s-step forward, can keep in u
        self.u, _ = torch.linalg.qr(self.k)
        self.u.requires_grad = False
        # aux_N -> aux_Unp1.T @ aux_U
        #   used in setting s,
        self.n = self.u.T @ self.unp1
        # ------- l postpro ----------------------------------
        # aux_Vtnp1 -> q from qr(lt.T)
        self.vt, _ = torch.linalg.qr(self.lt.T)
        # aux_M -> aux_Vtnp1 @ aux_Vt.T
        self.m = self.vt @ self.vtnp1.T
        # overwrite the old vars once there is are new ones
        self.unp1 = self.u
        self.vtnp1 = self.vt
        # ------- s prepro ------------------------------------
        # bias is trainable for the s step
        self.bias.requires_grad = True
        # set aux_U -> aux_Unp1  # done above
        # set aux_Vt -> aux_Vtnp1  # done previously now
        # set s -> (aux_N @ s) @ aux_M.T
        self.s = self.n @ self.s @ self.m.T


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
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if low_rank_percent is None:
            # set the max low_rank to be such that the
            roots = np.roots([1, in_features + out_features, in_features * out_features])
            pos_coeff = roots[roots > 0]
            if len(pos_coeff) < 1:
                self.rmax = min([in_features, out_features]) // 2
            else:
                self.rmax = int(np.floor(pos_coeff[-1]))
            # set the initial low_rank to be most of the rmax
            self.low_rank = self.rmax // 2
        else:
            self.rmax = min([in_features, out_features]) // 2
            self.low_rank = int(self.rmax * low_rank_percent)
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

        self.reset_parameters()
        self.train_case = "k"

        self.set_aux_vars()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, low_rank={self.low_rank}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )

    def set_aux_vars(self):
        factory = {"device": self.k.device, "dtype": self.k.dtype}
        with torch.no_grad():
            lr = self.low_rank
            lr2 = 2 * lr
            # needed at top: u, vt
            self.u = torch.empty((self.in_features, self.rmax), requires_grad=False, **factory)
            nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
            self.unp1 = torch.empty((self.in_features, 2 * self.rmax), requires_grad=False, **factory)
            nn.init.kaiming_uniform_(self.unp1, a=math.sqrt(5))
            self.vt = torch.empty((self.rmax, self.out_features), requires_grad=False, **factory)
            nn.init.kaiming_uniform_(self.m, a=math.sqrt(5))
            self.vtnp1 = torch.empty(
                (2 * self.rmax, self.out_features),
                requires_grad=False,
                **factory,
            )
            nn.init.kaiming_uniform_(self.vtnp1, a=math.sqrt(5))
            self.n = torch.empty((2 * self.rmax, self.rmax), requires_grad=False, **factory)
            self.m = torch.empty((2 * self.rmax, self.rmax), requires_grad=False, **factory)
            nn.init.kaiming_uniform_(self.n, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.m, a=math.sqrt(5))
            # need to check sizes...
            # from k postpro ---------------------------------------------------
            k_extended = torch.cat((self.k[:, :lr], self.u[:, :lr]), dim=1)
            prev_u, _ = torch.linalg.qr(k_extended)
            # aux_N -> aux_Unp1.T @ aux_U
            self.unp1[:, :lr2] = prev_u
            self.unp1.requires_grad = False
            self.u.requires_grad = False
            #   used in setting s,
            self.n[: 2 * lr, :lr] = self.unp1[:, :lr2].T @ self.u[:, :lr]
            self.n.requires_grad = False
            # from l postpro ---------------------------------------------------
            l_extended = torch.cat((self.lt[:lr].T, self.vt[:lr]), dim=1)
            aux_Vnp1, _ = torch.linalg.qr(l_extended)
            self.vtnp1[:lr2] = aux_Vnp1.T
            self.m[:lr2, :lr] = self.vtnp1[:lr2, :] @ self.vt[:lr].T
            self.vtnp1.requires_grad = False
            self.m.requires_grad = False

            # from rank adjustment ---------------------------------------------------
            s_small = self.s[: 2 * self.low_rank, : 2 * self.low_rank]
            u2, sing, vh2 = torch.linalg.svd(s_small, full_matrices=False)
            v2 = vh2.T

            rmax = self.rmax
            # update s
            self.s[:rmax, :rmax] = torch.diag(sing[:rmax], **factory)

            # create and update u and v
            self.u[:, :rmax] = self.unp1[:, : 2 * self.low_rank] @ u2[:, :rmax]
            self.vt[:rmax, :] = v2[:rmax, :] @ self.vtnp1[: 2 * self.low_rank, :]
            self.u.requires_grad = False
            self.vt.requires_grad = False
            # self.low_rank = int(rmax)

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
            self.lt += ltt.T[:, : self.low_rank].clone().detach()
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

    def forward(self, input: Tensor) -> Tensor:
        if self.train_case == "k":  # k-step
            ret = (input @ self.k[:, : self.low_rank]) @ self.vt[: self.low_rank]
        elif self.train_case == "l":  # l-step
            ret = (input @ self.u[:, : self.low_rank]) @ self.lt[: self.low_rank]
        else:  # s-step
            lr2 = 2 * self.low_rank
            ret = ((input @ self.unp1[:, :lr2]) @ self.s[:lr2, :lr2]) @ self.vtnp1[:lr2]

        return ret if self.bias is None else ret + self.bias

    @torch.no_grad()
    def kl_prepro(self):
        # disable bias training
        self.bias.requires_grad = False
        # k prepro
        # k -> aux_U @ s
        kls = self.s[: self.low_rank, : self.low_rank]
        self.k[:, : self.low_rank] = self.u[:, : self.low_rank] @ kls
        # l prepro
        # lt -> s @ aux_Vt
        self.lt[: self.low_rank] = kls @ self.vt[: self.low_rank]

    @torch.no_grad()
    def kl_postpro_s_prepro(self):
        lr = self.low_rank
        lr2 = 2 * self.low_rank
        # ------- k postpro ----------------------------------
        # aux_Unp1 -> q from qr(k)
        #   aux_Unp1 used in s-step forward, can keep in u
        k_extended = torch.cat((self.k[:, :lr], self.u[:, :lr]), dim=1)
        prev_u, _ = torch.linalg.qr(k_extended)
        # aux_N -> aux_Unp1.T @ aux_U
        self.unp1[:, :lr2] = prev_u
        self.unp1.requires_grad = False
        self.u.requires_grad = False
        #   used in setting s,
        aux_n = self.unp1[:, :lr2].T @ self.u[:, :lr]
        self.n[: 2 * lr, :lr] = aux_n
        self.n.requires_grad = False
        # ------- l postpro ----------------------------------
        l_extended = torch.cat((self.lt[:lr].T, self.vt[:lr]), dim=1)
        aux_Vnp1, _ = torch.linalg.qr(l_extended)
        self.vtnp1[:lr2] = aux_Vnp1.T
        m = self.vtnp1[:lr2, :] @ self.vt[:lr].T
        self.m[:lr2, :lr] = m
        self.vtnp1.requires_grad = False
        self.m.requires_grad = False
        # ------- s prepro ------------------------------------
        # bias is trainable for the s step
        self.bias.requires_grad = True
        # set s -> (aux_N @ s) @ aux_M.T
        self.s[:lr2, :lr2] = self.n[:lr2, :lr] @ self.s[:lr, :lr] @ self.m[:lr2, :lr].T

    def rank_adaption(self):
        # 1) compute SVD of S
        # d=singular values, u2 = left singuar vecs, v2= right singular vecs
        s_small = self.s[: 2 * self.low_rank, : 2 * self.low_rank]
        u2, sing, vh2 = torch.linalg.svd(s_small, full_matrices=False)
        v2 = vh2.T
        # d, u2, v2 = tf.linalg.svd(s_small)

        # absolute value treshold (try also relative one)
        tol = self.eps_adapt * torch.linalg.norm(sing)
        rmax = sing.shape[0] // 2
        for j in range(2 * rmax - 1):
            tmp = torch.linalg.norm(sing[j : 2 * rmax - 1])
            if tmp < tol:
                rmax = j
                break

        factory = {"dtype": rmax.dtype, "device": rmax.device}
        rmax = torch.minimum(rmax, self.rmax)
        rmax = torch.maximum(rmax, torch.full(tuple(rmax.shape), 2, **factory))

        # update s
        # self.s[:rmax, :rmax] = torch.diag(sing[:rmax], **factory)
        self.s = torch.diag(sing[:rmax], **factory)

        # update u and v
        # self.u[:, :rmax] = self.unp1[:, : 2 * self.low_rank] @ u2[:, :rmax]
        # self.vt[:rmax, :] = v2[:rmax, :] @ self.vtnp1[: 2 * self.low_rank, :]
        self.u = self.unp1[:, : 2 * self.low_rank] @ u2[:, :rmax]
        self.vt = v2[:rmax, :] @ self.vtnp1[: 2 * self.low_rank, :]
        self.low_rank = int(rmax)
