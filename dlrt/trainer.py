from __future__ import annotations

import time
from collections import namedtuple

import torch
import torch.distributed as dist
import torch.nn as nn
from rich import print
from rich.columns import Columns

from .conv import DLRTConv2d
from .linear import DLRTLinear


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
        mixed_precision: bool = True,
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
        self.reset_layers = None
        self.model = self.replace_linear_layers(self.model)
        #self.model = self.replace_linear_layers(self.model)
        # TODO: fix the last layer issue...
        #self.model = self._reset_last_layer_to_dense(self.model)
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
            self.smodel = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        else:
            self.kmodel = self.model
            self.lmodel = self.model
            self.smodel = self.model

        self.rank = 0 if not dist.is_initialized() else dist.get_rank()
        # need to re-init the optimizer with the new DLRT parameters
        optimizer_kwargs["params"] = self.model.parameters()
        self.optimizer = getattr(torch.optim, optimizer_name)(**optimizer_kwargs)
        # todo: autocast
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        if mixed_precision:
            # TODO: should there be different scalers for different parameters?
            #       i.e. -> one for k, onr for s, one for l
            self.kscaler = torch.cuda.amp.GradScaler()
            self.lscaler = torch.cuda.amp.GradScaler()
            self.sscaler = torch.cuda.amp.GradScaler()
        else:
            self.kscaler = None
        self.return_tuple = namedtuple("Trainer", ["loss", "output"])
        self.counter = 0

    def replace_linear_layers(self, module, name=None, process_group=None):
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
        elif isinstance(module, nn.Conv1d):  # Conv2d):
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
            #print(f"replacing {name} old: {module} with {module_output}")
            #del module
            #module = module_output

        for name, child in module.named_children():
            #print(name, child.extra_repr())
            module_output.add_module(name, self.replace_linear_layers(child, name, process_group))
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

    def cycle_layers(self):
        self.__run_command_on_dlrt_layers(module=self.model, command="cycle_training_case")

    def _set_training_all_params(self, network, totrain):
        for n, m in network.named_parameters():
            m.requires_grad = totrain
            # print('k', n, m.requires_grad)

    def set_layer_case(self, case):
        models = [self.model]
        if case in ["k", "l"]:
            # self.model.eval()
            self._set_training_all_params(network=self.model, totrain=False)
            try:
                self._set_training_all_params(network=self.kmodel, totrain=False)
                self._set_training_all_params(network=self.lmodel, totrain=False)
                models.append(getattr(self, f"{case}model"))
            except AttributeError:
                pass
        else:  # s case -> train all layers
            self._set_training_all_params(network=self.model, totrain=True)
            # self.model.train()
            try:
                self._set_training_all_params(network=self.smodel, totrain=True)
                # self.smodel.train()
                models.append(self.smodel)
            except AttributeError:
                pass

        for m in models:
            self.__run_command_on_dlrt_layers(
                module=m,  # getattr(self, f"{case}model"),
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

    def __collect_ranks(self, module, name=None):
        if hasattr(module, "dlrt"):
            #lst = [name, None, None]

            #perc, rnk = module.get_rank_percentage()
            #lst[1] = perc
            #lst[2] = rnk
            self.ranks.append(f"{name} {module.get_rank_percentage()}")
            #self.ranks.append(lst)

        for name, child in module.named_children():
            self.__collect_ranks(child, name)

    def get_all_ranks(self):
        self.ranks = []
        self.__collect_ranks(self.model)
        out_ranks = self.ranks.copy()
        self.ranks = []
        return out_ranks

    def _run_model(self, inputs, labels, case):
        if self.kscaler is not None:  # only test for kscaler -> rest are not always defined
            scaler = getattr(self, f"{case}scaler")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = getattr(self, f"{case}model")(inputs)
                loss = self.criterion(output, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            output = getattr(self, f"{case}model")(inputs)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
        return loss, output

    def train_step(self, model_inputs, labels, adapt=True):
        self.optimizer.zero_grad()
        # K
        self.set_layer_case("k")
        self.run_preproces(case="k")
        kloss, kret = self._run_model(model_inputs, labels, case="k")

        self.optimizer.zero_grad()
        self.run_postproces(case="k")

        # L
        # inputs = inputs.detach()
        self.set_layer_case("l")
        self.run_preproces(case="l")
        lloss, lret = self._run_model(model_inputs, labels, case="l")

        self.optimizer.zero_grad()

        self.run_postproces(case="l")
        # end of no_sync
        # inputs = inputs.detach()
        # S
        self.set_layer_case("s")
        self.run_preproces(case="s")
        sloss, sret = self._run_model(model_inputs, labels, case="s")

        # todo: set up scheduler
        if self.adaptive and adapt:
            self.run_rank_adaption()

            if self.rank == 0 and self.counter % 100 == 0:
                columns = Columns(self.get_all_ranks(), equal=True, expand=True)
                print(columns)
        self.counter += 1

        return self.return_tuple(sloss, sret)

    @torch.no_grad()
    def valid_step(self, model_inputs, labels):
        self.set_layer_case("s")
        self.run_preproces(case="s")
        sret = self.model(model_inputs)
        ls = self.criterion(sret, labels)
        return self.return_tuple(ls, sret)
