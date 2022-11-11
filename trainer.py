from __future__ import annotations

import torch
import torch.nn as nn

from conv import DLRTConv2dAdaptive
from conv import DLRTConv2dFixed
from linear import DLRTLinearAdaptive
from linear import DLRTLinearFixed


class DLRTTrainer(nn.Module):
    # class which will wrap whole models
    def __init__(
        self,
        torch_model: nn.Module,
        optimizer_name: str,
        optimizer_kwargs: dict,
        loss_function,
        scheduler=None,  # TODO: implement
        rank_percent: float = None,
        adaptive: bool = True,
        mixed_precision: bool = False,
        init_method="random",
    ):
        if "lr" not in optimizer_kwargs.keys():
            raise ValueError("LR must be included in optimizer_kwargs")
        if rank_percent and rank_percent > 1:
            raise ValueError(
                f"rank_percent should be less than 1, but got rank_percent={rank_percent}",
            )
        super().__init__()
        self.adaptive = adaptive
        self.rank_percent = rank_percent
        self.init_method = init_method
        self.loss_fn = loss_function

        # replace linear layers
        self.model = torch_model
        self.model = self.replace_linear_layers(self.model)
        self.ranks = []
        # need to re-init the optimizer with the new DLRT parameters
        optimizer_kwargs["params"] = self.model.parameters()
        self.optimizer = getattr(torch.optim, optimizer_name)(**optimizer_kwargs)
        # todo: autocast
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision

    def replace_linear_layers(self, module, process_group=None):
        module_output = module
        # this will remove all the BatchNorm layers from the network
        if isinstance(module, nn.Linear):
            model_kwargs = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                # "device": module.weight.device,
                # "dtype": module.weight.dtype,
            }
            if self.adaptive:
                module_output = DLRTLinearAdaptive(
                    **model_kwargs,
                    low_rank_percent=self.rank_percent,
                )
            else:
                module_output = DLRTLinearFixed(
                    **model_kwargs,
                    low_rank=int(self.rank_percent * module.in_features * module.out_features),
                )
        elif isinstance(module, nn.Conv2d):
            model_kwargs = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "dilation": module.dilation,
                "groups": module.groups,
                "bias": module.bias,
                "padding_mode": module.padding_mode,
                # "device": module.device,
                # "dtype": module.dtype,
            }
            if self.adaptive:
                module_output = DLRTConv2dAdaptive(
                    **model_kwargs,
                    low_rank_percent=self.rank_percent,
                )
            else:
                module_output = DLRTConv2dFixed(**model_kwargs, low_rank_percent=self.rank_percent)

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

    def train_step(self, model_inputs, labels):
        self.optimizer.zero_grad()

        # K
        self.set_layer_case("k")
        self.run_preproces(case="k")
        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)
        # TODO: autocast model with AMP
        kret = self.model(model_inputs)
        kloss = self.loss_fn(kret, labels)
        kloss.backward()

        # optimizer
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.run_postproces(case="k")

        # L
        self.set_layer_case("l")
        self.run_preproces(case="l")
        lret = self.model(model_inputs)
        lloss = self.loss_fn(lret, labels)
        lloss.backward()
        # optimizer
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.run_postproces(case="l")

        # S
        self.set_layer_case("s")
        self.run_preproces(case="s")
        sret = self.model(model_inputs)
        sloss = self.loss_fn(sret, labels)
        sloss.backward()

        # optimizer
        self.optimizer.step()
        # todo: set up scheduler
        if self.adaptive:
            self.run_rank_adaption()

        print(self.get_all_ranks())

        return sloss

    @torch.no_grad()
    def valid_step(self, model_inputs, labels):
        self.set_layer_case("s")
        self.run_preproces(case="s")
        sret = self.model(model_inputs)
        ls = self.loss_fn(sret, labels)
        return ls, sret
