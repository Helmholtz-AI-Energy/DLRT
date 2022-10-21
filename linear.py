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
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.dlra = True
        self.init_method = init_method
        assert init_method in ["random", "svd"], "init_method must be in ['random', 'svd']"

        # need k, lt, s, bias
        # K -> U @ S, L -> V @ S.T
        self.k = nn.Parameter(torch.empty((in_features, rank), **factory_kwargs))
        self.s = nn.Parameter(torch.empty((rank, rank), **factory_kwargs))
        self.lt = nn.Parameter(torch.empty((rank, out_features), **factory_kwargs))

        self.reset_parameters()
        self.train_case = "k"

        with torch.no_grad():
            # u, vt, n, m, uprev, vtprev
            self.u, _ = torch.linalg.qr(self.k)
            self.u.requires_grad = False
            # aux_N -> aux_Unp1.T @ aux_U
            #   used in setting s,
            self.n = self.u.T @ self.uprev
            self.n.requires_grad = False
            # aux_Vtnp1 -> q from qr(lt.T)
            self.vt, _ = torch.linalg.qr(self.lt.T)
            self.vt.requires_grad = False
            # aux_M -> aux_Vtnp1 @ aux_Vt.T
            self.m = self.vt @ self.vtprev.T
            # overwrite the old vars once there is are new ones
            self.uprev = torch.empty(tuple(self.u.shape), **factory_kwargs)
            self.vtprev = torch.empty(tuple(self.vt.shape), **factory_kwargs)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, rank={self.rank}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )

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
            self.k += k[:, : self.rank].clone().detach()
            self.s *= 0
            self.s += torch.diag(vec_s[: self.rank].clone().detach())
            self.lt *= 0
            # TODO: transpose needed??
            self.lt += ltt.T[:, : self.rank].clone().detach()
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def change_training_case(self, case):
        # switch -> if current train case is k/l, do post for
        if self.train_case in "k":
            pass

    def forward(self, input: Tensor) -> Tensor:
        if self.train_case == "k":  # k-step
            z = torch.matmul(torch.matmul(input, self.k), self.aux_Vt) + self.aux_b
        elif self.train_case == "l":  # l-step
            z = torch.matmul(torch.matmul(input, self.aux_U), self.lt) + self.aux_b
        else:  # s-step
            z = (
                torch.matmul(torch.matmul(torch.matmul(input, self.aux_Unp1), self.s), self.aux_Vtnp1)
                + self.b
            )
        return z

    @torch.no_grad()
    def kl_prepro(self):
        # disable bias training
        self.bias.requires_grad = False
        # k prepro
        # k -> aux_U @ s (todo: s is trainable here?)
        self.k = self.u @ self.s
        # l prepro
        # lt -> s @ aux_Vt   (todo: s is trainable?)
        self.lt = self.s @ self.vt

    # @torch.no_grad()
    # def k_step_prepro(self):
    #     # set bias to not train
    #     self.bias.requires_grad = False
    #     # set k to be aux_U @ s (todo: s is trainable here?)
    #     #   aux_U is not trainable here
    # @torch.no_grad()
    # def l_step_prepro(self):
    #     # make sure bias is not training
    #     self.bias.requires_grad = False
    #     # lt = s @ aux_Vt   (todo: s is trainable?)
    #     pass

    @torch.no_grad()
    def kl_postpro_s_prepro(self):
        # ------- k postpro ----------------------------------
        # aux_Unp1 -> q from qr(k)
        #   aux_Unp1 used in s-step forward, can keep in u
        self.u, _ = torch.linalg.qr(self.k)
        self.u.requires_grad = False
        # aux_N -> aux_Unp1.T @ aux_U
        #   used in setting s,
        self.n = self.u.T @ self.uprev
        # ------- l postpro ----------------------------------
        # aux_Vtnp1 -> q from qr(lt.T)
        self.vt, _ = torch.linalg.qr(self.lt.T)
        # aux_M -> aux_Vtnp1 @ aux_Vt.T
        self.m = self.vt @ self.vtprev.T
        # overwrite the old vars once there is are new ones
        self.uprev = self.u
        self.vtprev = self.vt
        # ------- s prepro ------------------------------------
        # bias is trainable for the s step
        self.bias.requires_grad = True
        # set aux_U -> aux_Unp1  # done above
        # set aux_Vt -> aux_Vtnp1  # done previously now
        # set s -> (aux_N @ s) @ aux_M.T
        self.s = self.n @ self.s @ self.m.T

    # @torch.no_grad()
    # def k_step_postpro(self):
    #     # aux_Unp1 -> q from qr(k)
    #     # aux_N -> aux_Unp1.T @ aux_U
    #     pass
    # @torch.no_grad()
    # def l_step_postpro(self):
    #     # aux_Vtnp1 -> q from qr(lt.T)
    #     # aux_M -> aux_Vtnp1 @ aux_Vt.T
    #     pass
    # @torch.no_grad()
    # def s_step_prepro(self):
    #     # bias is trainable here
    #     self.bias.requires_grad = True
    #     # set aux_U -> aux_Unp1
    #     # set aux_Vt -> aux_Vtnp1
    #     # set s -> (aux_N @ s) @ aux_M.T
    #     pass
