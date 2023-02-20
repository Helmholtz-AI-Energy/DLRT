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

console = Console(width=140)

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
        ddp_dlrt_layers: bool = False,
        dense_first_layer: bool = False,
        dense_last_layer: bool = False,
        pretrain_count: int = 0,
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
        self.torch_model = torch_model
        self.reset_layers = None
        if not dist.is_initialized():
            self.ddp_dlrt_layers = False
            self.rank = 0
        else:
            self.ddp_dlrt_layers = ddp_dlrt_layers
            self.rank = dist.get_rank()

        self.pretrain_count = pretrain_count
        self.in_pretrain = lambda: self.pretrain_count > 0
        self.dense_last_layer = dense_last_layer
        self.dense_first_layer = dense_first_layer
        self._dfl_wait = dense_first_layer

        self.current_layer_train_case = "pretrain" if self.in_pretrain() else "k"
        self.wrap_model()

    @torch.no_grad()
    def wrap_model(self):
        self.first_layer = None
        self.dlrt_model = self._replace_layers(
            self.torch_model,
            pretrain=self.in_pretrain(),
        )
        if self.dense_last_layer:
            self.dlrt_model = self._reset_last_layer_to_dense(self.dlrt_model)

        self.__run_command_on_dlrt_layers(
            module=self.dlrt_model,
            command="set_dlrt_requires_grad",
            kwargs={"requires": False},
        )

        if dist.is_initialized():
            # need a seperate DDP instance for each training case, only way to have diff buckets
            # self.pretrainmodel = torch.nn.parallel.DistributedDataParallel(
            #     self.dlrt_model,
            #     find_unused_parameters=False,
            # )

            self.set_layer_case("k")
            self.run_preprocess(case="k")
            self.kmodel = torch.nn.parallel.DistributedDataParallel(
                self.dlrt_model,
                find_unused_parameters=True,
            )
            self.lmodel = self.kmodel
            self.smodel = self.kmodel
            self.pretrainmodel = self.kmodel
            # self.set_layer_case("l")
            # self.run_preprocess(case="l")
            # # self.lmodel = torch.nn.parallel.DistributedDataParallel(
            # #     self.dlrt_model,
            # #     find_unused_parameters=False,
            # # )
            # self.set_layer_case("s")
            # self.run_preprocess(case="s")
            # self.smodel = torch.nn.parallel.DistributedDataParallel(
            #     self.dlrt_model,
            #     find_unused_parameters=False,
            # )
            for layer in self.dlrt_model.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        else:
            # pass
            self.pretrainmodel = self.dlrt_model
            self.kmodel = self.dlrt_model
            self.lmodel = self.dlrt_model
            self.smodel = self.dlrt_model

    def _replace_layers(self, module, pretrain=False, name=None, process_group=None):
        module_output = module
        # this will remove all the BatchNorm layers from the network
        # TODO: add warning that the replaced layers are slower than CUDnn (but that is expected)
        if isinstance(module, nn.Linear):
            if not self._dfl_wait:  # if not waiting i.e. already past the first layer
                module_output = DLRTLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    adaptive=self.adaptive,
                    low_rank_percent=self.rank_percent,
                    eps_adapt=self.epsilon["linear"],
                    pretrain=pretrain,
                ).to(device=module.weight.device, dtype=module.weight.dtype)
                self.reset_layers = [module, name]
            else:  # dont wait -> is first layer -> should be dense
                self._dfl_wait = False
        elif isinstance(module, nn.Conv2d):
            if not self._dfl_wait:  # if not waiting i.e. already past the first layer
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
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    eps_adapt=self.epsilon["conv2d"],
                    pretrain=pretrain,
                ).to(device=module.weight.device, dtype=module.weight.dtype)
                self.reset_layers = [module, name]
                # del module
            else:  # dont wait -> is first layer -> should be dense
                self._dfl_wait = False

        for name, child in module.named_children():
            module_output.add_module(
                name,
                self._replace_layers(
                    child,
                    pretrain=pretrain,
                    name=name,
                    process_group=process_group,
                ),
            )
        del module
        return module_output

    def _reset_last_layer_to_dense(self, module, name=None):
        # if dist.get_rank() == 0:
        #     print("replace", name)
        module_output = module
        if name == self.reset_layers[1]:
            if hasattr(module, "weight"):
                device = module.weight.device
                dtype = module.weight.dtype
            else:
                try:
                    device = module.k.device
                    dtype = module.k.dtype
                except AttributeError:
                    device = None
            if device is not None:
                module_output = self.reset_layers[0].to(device=device, dtype=dtype)
        for name, child in module.named_children():
            module_output.add_module(name, self._reset_last_layer_to_dense(child, name))
        # del module
        return module_output

    def _set_training_all_params(self, network, totrain):
        for n, m in network.named_parameters():
            m.requires_grad = totrain
            m.training = True  # totrain
            try:
                m.track_running_stats = totrain
            except AttributeError:
                pass

    def set_layer_case(self, case):
        # set the training case of all DLRT layers (conv/linear)
        models = [self.dlrt_model]
        # self.optimizer.zero_grad(set_to_none=True)
        # self.dlrt_model.train()
        if case == "k":
            try:
                models.append(getattr(self, "kmodel"))
                # self._set_training_all_params(network=self.kmodel, totrain=True)
            except AttributeError:
                pass
        if case == "l":
            # turn off training on all layers
            # self.dlrt_model.eval()
            # self._set_training_all_params(network=self.dlrt_model, totrain=False)
            try:
                models.append(getattr(self, "kmodel"))
                # self._set_training_all_params(network=self.lmodel, totrain=False)
            except AttributeError:
                pass
        else:  # s case -> train all layers, turn off training of K and L
            # self.dlrt_model.train()
            # self._set_training_all_params(network=self.dlrt_model, totrain=True)
            try:
                # self.smodel.train()
                # self._set_training_all_params(network=self.smodel, totrain=False)
                models.append(self.smodel)
            except AttributeError:
                pass

        for m in models:
            self.__run_command_on_dlrt_layers(
                module=m,
                command="change_training_case",
                kwargs={"case": case},
            )
            # self.__run_command_on_dlrt_layers(module=m, command="train")

    def run_preprocess(self, case):
        # prev: getattr(self, f"{case}model")
        self.__run_command_on_dlrt_layers(module=self.dlrt_model, command=f"{case}_preprocess")

    def run_postprocess(self, case):
        self.__run_command_on_dlrt_layers(module=self.dlrt_model, command=f"{case}_postprocess")

    def run_rank_adaption(self, skip=False, all_reduce_method="average"):
        self.__run_command_on_dlrt_layers(
            module=self.dlrt_model,
            command="rank_adaption",
            kwargs={"skip": skip},
        )
        # self.__run_command_on_dlrt_layers(
        #     module=self.dlrt_model, command="all_reduce", kwargs={"method": all_reduce_method}
        # )

    def stop_pretraining(self):
        self.__run_command_on_dlrt_layers(module=self.dlrt_model, command="stop_pretraining")

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        # self.dlrt_model.training = mode
        self.dlrt_model.training = mode
        # self._train(self.dlrt_model, mode)
        for module in self.dlrt_model.children():
            # print(module)
            module.train(mode)
        # todo: recursse deeper...?
        # TODO: fix me in DDP?? (do this on k/l/s models?)
        # self.dlrt_model = self.dlrt_model.train()
        return self

    def _train(self, module, mode=True):
        module.train(mode)
        # module.training = mode
        for n, child in module.named_children():
            # print(n, child.training)
            # child.train(mode)
            self._train(child)

    def eval(self):
        return self.train(False)

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
        self.__collect_ranks(self.dlrt_model)
        out_ranks = self.ranks.copy()
        self.ranks = []
        return out_ranks

    def __call__(self, inputs, case):
        # if pretrain:
        #     return self.premodel(inputs)
        # if case != self.current_layer_train_case:
        #     self.set_layer_case(case=case)
        with getattr(self, f"{case}model").no_sync():
            return getattr(self, f"{case}model")(inputs)
