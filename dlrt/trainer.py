from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn
import torch.distributed as dist

from .conv import DLRTConv2d
from .linear import DLRTLinear

import time


__all__ = ["DLRTTrainer"]


class DLRTTrainer:
    # class which will wrap whole models
    def __init__(
        self,
        torch_model: nn.Module,
        optimizer_name: str,
        optimizer_kwargs: dict,
        criterion,
        scheduler=None,  # TODO: implement
        rank_percent: float = None,
        adaptive: bool = True,
        mixed_precision: bool = False,
        init_method="random",
        epsilon=None,
        init_ddp: bool = False,
    ):
        if epsilon is None:
            epsilon = {"linear": 0.1, "conv2d": 0.1}
        elif not isinstance(epsilon, dict):
            raise TypeError(
                f"epsilon must be a dict with a value for every type of DLRT layer ('linear, "
                f"conv2d', transformers), currently: {epsilon}",
            )
        if "lr" not in optimizer_kwargs.keys():
            raise ValueError("LR must be included in optimizer_kwargs")
        if rank_percent and rank_percent > 1:
            raise ValueError(
                f"rank_percent should be less than 1, but got rank_percent={rank_percent}",
            )
        super().__init__()
        self.adaptive = adaptive
        self.rank_percent = rank_percent
        self.epsilon = epsilon
        self.init_method = init_method
        self.criterion = criterion

        # replace linear layers
        self.model = torch_model
        self.model = self.replace_linear_layers(self.model)

        if init_ddp:
            torch.nn.parallel.DistributedDataParallel(self.model)  # , device_ids=[args.gpu])

        self.ranks = []
        # need to re-init the optimizer with the new DLRT parameters
        optimizer_kwargs["params"] = self.model.parameters()
        self.optimizer = getattr(torch.optim, optimizer_name)(**optimizer_kwargs)
        # todo: autocast
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.return_tuple = namedtuple("Trainer", ["loss", "output"])

    def replace_linear_layers(self, module, process_group=None):
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
        elif isinstance(module, nn.Conv2d):
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

        for name, child in module.named_children():
            module_output.add_module(name, self.replace_linear_layers(child, process_group))
        del module
        return module_output

    def cycle_layers(self):
        self.__run_command_on_dlrt_layers(module=self.model, command="cycle_training_case")

    def set_layer_case(self, case):
        if case in ["k", "l"]:
            self.model.eval()
        else:  # s case -> train all layers
            self.model.train()
        self.__run_command_on_dlrt_layers(
            module=self.model,
            command="change_training_case",
            kwargs={"case": case},
        )

    def run_preproces(self, case):
        self.__run_command_on_dlrt_layers(module=self.model, command=f"{case}_preprocess")

    def run_postproces(self, case):
        self.__run_command_on_dlrt_layers(module=self.model, command=f"{case}_postprocess")

    def run_rank_adaption(self):
        self.__run_command_on_dlrt_layers(module=self.model, command="rank_adaption")

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __run_command_on_dlrt_layers(self, module, command, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if hasattr(module, "dlrt"):
            getattr(module, command)(**kwargs)

        for name, child in module.named_children():
            self.__run_command_on_dlrt_layers(child, command, kwargs)

    def __collect_ranks(self, module):
        if hasattr(module, "dlrt"):
            self.ranks.append(module.get_rank_percentage())

        for name, child in module.named_children():
            self.__collect_ranks(child)

    def get_all_ranks(self):
        self.ranks = []
        self.__collect_ranks(self.model)
        out_ranks = self.ranks.copy()
        self.ranks = []
        return out_ranks

    def train_step(self, model_inputs, labels, adapt=True):
        self.optimizer.zero_grad()

        # K
        self.set_layer_case("k")
        self.run_preproces(case="k")
        #if dist.is_initialized() and dist.get_rank() == 0:
        #    c = 0
        #    for name, param in self.model.named_parameters():
        #        if name == "conv1.l":
        #            k_test = param
        #            #print(name, param[..., :10])
        #        #c += 1
        #        #if c == 10:
        #        #    break
        # raise ValueError("")
        # TODO: autocast model with AMP
        kret = self.model(model_inputs)
        kloss = self.criterion(kret, labels)
        t1 = time.perf_counter()
        kloss.backward()

        # optimizer
        self.optimizer.step()
        #if dist.get_rank() == 0:
        #    print(f"K-backwards time: {time.perf_counter() - t1}")
        #    #if dist.is_initialized() and dist.get_rank() == 0:
        #    #c = 0
        #    for name, param in self.model.named_parameters():
        #        if name == "conv1.l":
        #            print("L test (should ALWAYS be true)", torch.equal(k_test, param))
        #            #print(name, param[..., :10])
        self.optimizer.zero_grad()

        self.run_postproces(case="k")

        # L
        self.set_layer_case("l")
        self.run_preproces(case="l")
        lret = self.model(model_inputs)
        lloss = self.criterion(lret, labels)
        t2 = time.perf_counter()
        lloss.backward()
        # optimizer
        self.optimizer.step()
        #if dist.get_rank() == 0:
        #    print(f"L-backwards time: {time.perf_counter() - t2}")

        self.optimizer.zero_grad()

        self.run_postproces(case="l")

        # S
        self.set_layer_case("s")
        self.run_preproces(case="s")
        sret = self.model(model_inputs)
        sloss = self.criterion(sret, labels)
        t3 = time.perf_counter()
        sloss.backward()

        # optimizer
        self.optimizer.step()
        #if dist.get_rank() == 0:
        #    print(f"S-backwards time: {time.perf_counter() - t3}")

        # todo: set up scheduler
        if self.adaptive and adapt:
            self.run_rank_adaption()

        #if dist.get_rank() == 0:
        #    print(self.get_all_ranks())

        return self.return_tuple(sloss, sret)

    @torch.no_grad()
    def valid_step(self, model_inputs, labels):
        self.set_layer_case("s")
        self.run_preproces(case="s")
        sret = self.model(model_inputs)
        ls = self.criterion(sret, labels)
        return self.return_tuple(ls, sret)
