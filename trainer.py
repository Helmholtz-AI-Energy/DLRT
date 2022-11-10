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
        optimizer: torch.optim.Optimizer,
        loss_function,
        rank_percent: float = None,
        adaptive: bool = True,
        init_method="random",
    ):
        if rank_percent > 1:
            raise ValueError(
                f"rank_percent should be less than 1, but got rank_percent={rank_percent}",
            )
        super().__init__()
        self.apaptive = adaptive
        self.rank_percent = rank_percent
        self.init_method = init_method
        self.optimizer = optimizer
        self.loss_fn = loss_function

        # replace linear layers
        self.model = torch_model
        self.model = self.replace_linear_layers(self.model)

    def replace_linear_layers(self, module, process_group=None):
        module_output = module
        # this will remove all the BatchNorm layers from the network
        if isinstance(module, nn.Linear):
            model_kwargs = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                "device": module.weight.device,
                "dtype": module.weight.dtype,
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
                "device": module.device,
                "dtype": module.dtype,
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

    def run_klprepro(self):
        self.__run_command_on_dlrt_layers(module=self.model, command="kl_prepro")

    def run_kl_postpro_s_prepro(self):
        self.__run_command_on_dlrt_layers(module=self.model, command="kl_postpro_s_prepro")

    def __run_command_on_dlrt_layers(self, module, command, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if hasattr(module, "dlrt"):
            getattr(module, command)(**kwargs)

        for name, child in module.named_children():
            self.__run_command_on_dlrt_layers(child, command, kwargs)

    def train_step(self, model_inputs, labels):
        self.optimizer.zero_grad()

        self.run_klprepro()
        # K
        self.set_layer_case("k")
        # TODO: autocast model with AMP
        kret = self.model(model_inputs)
        kloss = self.loss_fn(kret, labels)
        kloss.backward()
        # L
        self.set_layer_case("l")
        lret = self.model(model_inputs)
        lloss = self.loss_fn(lret, labels)
        lloss.backward()

        # optimizer
        self.optimizer.step()
        self.optimizer.zero_grad()

        # S
        self.run_kl_postpro_s_prepro()

        self.set_layer_case("s")
        sret = self.model(model_inputs)
        sloss = self.loss_fn(sret, labels)
        sloss.backward()

        # optimizer
        self.optimizer.step()

        # postpro
        # self._cycle_layers(self.model)
        # self.model.forward(x)
        return sloss
