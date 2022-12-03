from __future__ import annotations

import time
from collections import namedtuple

import torch
import torch.distributed as dist
import torch.nn as nn
from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.pretty import Pretty

from .conv import DLRTConv2d
from .linear import DLRTLinear

console = Console(width=120)

__all__ = ["DLRTNetwork"]


class DLRTNetwork(nn.Module):
    # abstraction of a wrapped torch network. Thiw will be used to call the functions for all the
    # layers. it will hold things which dont need to be in the trainer class
    def __init__(
        self,
        torch_model: nn.Module,
        rank_percent: float = None,
        adaptive: bool = True,
        epsilon: float = None,
        init_ddp: bool = dist.is_initialized(),
        dense_last_layer: bool = False,
    ):
        super().__init__()
        self.adaptive = adaptive
        if epsilon is None:
            epsilon = {"linear": 0.1, "conv2d": 0.1}
        elif not isinstance(epsilon, dict):
            raise TypeError(
                f"epsilon must be a dict with a value for every type of DLRT layer ('linear, "
                f"conv2d', transformers), currently: {epsilon}",
            )
        if rank_percent and rank_percent > 1:
            raise ValueError(
                f"rank_percent should be less than 1, but got rank_percent={rank_percent}",
            )
        super().__init__()
        if not adaptive and rank_percent is None:
            raise ValueError("must have either adaptive or rank_percent")
        self.adaptive = adaptive
        self.rank_percent = rank_percent
        self.epsilon = epsilon

        # replace linear layers
        self.model = torch_model
        self.reset_layers = None
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

        self.model = self._replace_layers(self.model)
        if dense_last_layer:
            self.model = self._reset_last_layer_to_dense(self.model)
        if init_ddp:
            # need a seperate DDP instance for each training case, only way to have diff buckets
            self.set_layer_case("k")
            self.run_preproces(case="k")
            self.kmodel = torch.nn.parallel.DistributedDataParallel(self.model)
            self.set_layer_case("l")
            self.run_preproces(case="l")
            self.lmodel = torch.nn.parallel.DistributedDataParallel(self.model)
            self.set_layer_case("s")
            self.run_preproces(case="s")
            self.smodel = torch.nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=True,
            )
        else:
            # pass
            self.kmodel = self.model
            self.lmodel = self.model
            self.smodel = self.model

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        # this will remove all the BatchNorm layers from the network
        if isinstance(module, nn.Linear):
            module_output = DLRTLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                adaptive=self.adaptive,
                low_rank_percent=self.rank_percent,
                eps_adapt=self.epsilon["linear"],
                # TODO: device checks??
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            self.reset_layers = [module, name]
        elif isinstance(module, nn.Conv2d):
            # TODO: add warning that the replaced layers are slower than CUDnn (but that is
            #  expected)
            module_output = DLRTConv2d(
                adaptive=self.adaptive,
                low_rank_percent=self.rank_percent,
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias,
                padding_mode=module.padding_mode,
                eps_adapt=self.epsilon["conv2d"],
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            self.reset_layers = [module, name]
            # print(f"replacing {name} old: {module} with {module_output}")
            # del module
            # module = module_output

        for name, child in module.named_children():
            # print(name, child.extra_repr())
            module_output.add_module(name, self._replace_layers(child, name, process_group))
        del module
        return module_output

    def _reset_last_layer_to_dense(self, module, name=None):
        module_output = module
        # this will remove all the BatchNorm layers from the network
        if name == self.reset_layers[1]:
            if hasattr(module, "weight"):
                device = module.weight.device
                dtype = module.weight.dtype
            else:
                device = module.k.device
                dtype = module.k.dtype
            module_output = self.reset_layers[0].to(device=device, dtype=dtype)
        for name, child in module.named_children():
            module_output.add_module(name, self._reset_last_layer_to_dense(child, name))
        del module
        return module_output

    def _set_training_all_params(self, network, totrain):
        for n, m in network.named_parameters():
            m.requires_grad = totrain
            m.training = totrain
            try:
                m.track_running_stats = totrain
            except AttributeError:
                pass

    def set_layer_case(self, case):
        # set the training case of all DLRT layers (conv/linear)
        models = [self.model]
        # self.optimizer.zero_grad(set_to_none=True)
        if case in ["k", "l"]:
            # turn off training on all layers
            self._set_training_all_params(network=self.model, totrain=False)
            self.model.eval()
            try:
                self._set_training_all_params(network=self.kmodel, totrain=False)
                self._set_training_all_params(network=self.lmodel, totrain=False)
                models.append(getattr(self, f"{case}model"))
                self.kmodel.eval()
                self.lmodel.eval()
            except AttributeError:
                pass
        else:  # s case -> train all layers, turn off training of K and L
            self.model.train()
            self._set_training_all_params(network=self.model, totrain=True)
            try:
                self.smodel.train()
                self._set_training_all_params(network=self.smodel, totrain=True)
                # self.smodel.train()
                models.append(self.smodel)
            except AttributeError:
                pass

        for m in models:
            self.__run_command_on_dlrt_layers(
                module=m,
                command="change_training_case",
                kwargs={"case": case},
            )

    def run_preprocess(self, case):
        # prev: getattr(self, f"{case}model")
        self.__run_command_on_dlrt_layers(module=self.model, command=f"{case}_preprocess")

    def run_postprocess(self, case):
        self.__run_command_on_dlrt_layers(module=self.model, command=f"{case}_postprocess")

    def run_rank_adaption(self, skip=False):
        self.__run_command_on_dlrt_layers(module=self.model, command="rank_adaption", kwargs={"skip": skip})

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.model.training = mode
        for module in self.model.children():
            module.train(mode)
        # TODO: fix me in DDP?? (do this on k/l/s models?)
        # self.model = self.model.train()
        return self

    def eval(self):
        self = self.train(False)

    def __run_command_on_dlrt_layers(self, module, command, kwargs=None):
        # NOTE: the command must be a member function of DLRTModule
        if kwargs is None:
            kwargs = {}

        if hasattr(module, "dlrt"):
            getattr(module, command)(**kwargs)

        for name, child in module.named_children():
            self.__run_command_on_dlrt_layers(child, command, kwargs)

    def __collect_ranks(self, module, name=None):
        if hasattr(module, "dlrt"):
            # lst = [name, None, None]

            # perc, rnk = module.get_rank_percentage()
            # lst[1] = perc
            # lst[2] = rnk
            self.ranks.append(f"{name} {module.get_rank_percentage()}")
            # self.ranks.append(lst)

        for name, child in module.named_children():
            self.__collect_ranks(child, name)

    def get_all_ranks(self):
        self.ranks = []
        self.__collect_ranks(self.model)
        out_ranks = self.ranks.copy()
        self.ranks = []
        return out_ranks

    def __call__(self, inputs, case):
        return getattr(self, f"{case}model")(inputs)
