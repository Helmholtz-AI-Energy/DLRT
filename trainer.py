from __future__ import annotations

import torch
import torch.nn as nn

from linear import DLRALinear
from linear import DLRALinearAdaptive


class DLRATrainer(nn.Module):
    # class which will wrap whole models
    def __init__(self, torch_model, rank, init_method="random"):
        super().__init__()

        self.rank = rank
        self.init_method = init_method

        # replace linear layers
        self.model = torch_model
        self.model = self.replace_linear_layers(self.model)

    def replace_linear_layers(self, module, process_group=None):
        module_output = module
        # this will remove all the BatchNorm layers from the network
        if isinstance(module, nn.Linear):
            module_output = DLRALinearAdaptive(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
                # DLRA params
                low_rank_percent=None,
            )

        for name, child in module.named_children():
            module_output.add_module(name, self.replace_linear_layers(child, process_group))
        del module
        return module_output

    def _cycle_layers(self, module):
        # this will remove all the BatchNorm layers from the network
        try:
            module.cycle_training_case()
        except AttributeError:
            pass

        for name, child in module.named_children():
            self.cycle_layers(child)

    def forward(self, x):
        self._cycle_layers(self.model)
        return self.model.forward(x)
