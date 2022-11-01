from __future__ import annotations

import torch
import torch.nn as nn

from linear import DLRALinear


class DLRANet(nn.Module):
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
            module_output = DLRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
                # DLRA params
                init_method=self.init_method,
                rank=self.rank,
            )

        for name, child in module.named_children():
            module_output.add_module(name, self.replace_linear_layers(child, process_group))
        del module
        return module_output

    def forward(self, x):
        return self.model.forward(x)
