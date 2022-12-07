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

from .network import DLRTNetwork

console = Console(width=120)

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

        self.model = DLRTNetwork(
            torch_model=torch_model,
            rank_percent=rank_percent,
            adaptive=adaptive,
            epsilon=epsilon,
            dense_last_layer=False,
            init_ddp=init_ddp,
        )

        # need to rinit the optimizer with the new DLRT parameters
        optimizer_kwargs["params"] = self.model.parameters()
        self.optimizer = getattr(torch.optim, optimizer_name)(**optimizer_kwargs)
        print(Pretty({"Optimizer": optimizer_name, **optimizer_kwargs}))
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
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def _run_model(self, inputs, labels, case):
        #self.optimizer.zero_grad(set_to_none=True)

        if self.kscaler is not None:  # only test for kscaler -> rest are not always defined
            scaler = self.kscaler  # getattr(self, f"{case}scaler")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(inputs, case)
                loss = self.criterion(output, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            # scaler.step(self.optimizer)
            # scaler.update()
        else:
            output = self.model(inputs, case)
            loss = self.criterion(output, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        if torch.isnan(loss):
            raise RuntimeError(f"loss is NaN in case {case}! {output}")
        class_loss = getattr(self, f"{case}loss")
        class_loss *= 0.0
        class_loss += loss
        # TODO: remove output ???
        self.output = output
        return loss, output

    def train_step(self, inputs, labels, skip_adapt=False):
        #console.rule("top of train step")
        # TODO: remove this after debug...
        fact = {"device": inputs.device, "dtype": inputs.dtype}
        self.kloss, self.lloss, self.sloss = (
            torch.tensor(0, **fact),
            torch.tensor(0, **fact),
            torch.tensor(0, **fact),
        )
        self.output = None
        self.optimizer.zero_grad()
        for case in ["k", "l"]:
            self.model.set_layer_case(case)
            # self.model.run_preprocess(case)
            requires_grad = []
            self._run_model(inputs, labels, case)
            #for n, m in self.model.model.named_parameters():
            #    if n.startswith("conv"):  # m.requires_grad:
            #        requires_grad.append(
            #            f"{n}, {m.max().item():.4f}, {m.min().item():.4f}, {m.mean().item():.4f}, {m.std().item():.4f}, {m.requires_grad}",
            #        )
            #if self.rank == 0:# and self.counter % 100 == 0:
            #    columns = Columns(requires_grad, equal=True, expand=True)
            #    console.rule(f"After {case}")
            #    console.print(columns)
            #self._run_model(inputs, labels, case)
        if self.kscaler is not None:
            self.kscaler.step(self.optimizer)
            self.kscaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

        self.model.run_postprocess("k")
        self.model.run_postprocess("l")

        requires_grad = []
        for n, m in self.model.model.named_parameters():
            if n.startswith("fc"):  # m.requires_grad:
                requires_grad.append(
                    f"{n}, {m.max().item():.4f}, {m.min().item():.4f}, {m.mean().item():.4f}, {m.std().item():.4f}, {m.requires_grad}",
                )
        if self.rank == 0:# and self.counter % 100 == 0:
            columns = Columns(requires_grad, equal=True, expand=True)
            #console.rule("After k/l steps")
            #console.print(columns)

        self.model.set_layer_case("s")

        self.optimizer.zero_grad()
        #            "{m.mean().item():.4f}, {m.std().item():.4f}, {m.requires_grad}"
        #        )
        #if self.rank == 0: # and self.counter % 100 == 0:
        #    columns = Columns(requires_grad, equal=True, expand=True)
        #    console.rule("Before s")
        #    console.print(columns)

        self._run_model(inputs, labels, "s")
        if self.kscaler is not None:
            self.kscaler.step(self.optimizer)
            self.kscaler.update()
        else:
            self.optimizer.step()

        if self.adaptive:
            self.model.run_rank_adaption(skip_adapt)

            if self.rank == 0 and self.counter % 100 == 0 and not skip_adapt:
                console.rule("After rank adaptation")
                columns = Columns(self.model.get_all_ranks(), equal=True, expand=True)
                console.print(columns)

        self.counter += 1
        requires_grad = []
        for n, m in self.model.model.named_parameters():
            if n.startswith("fc"):  # m.requires_grad:
                requires_grad.append(
                    f"{n}, {m.max().item():.4f}, {m.min().item():.4f}, {m.mean().item():.4f}, {m.std().item():.4f}, {m.requires_grad}",
                )
        if self.rank == 0:# and self.counter % 100 == 0:
            columns = Columns(requires_grad, equal=True, expand=True)
            #console.rule(f"After s train step")
            #console.print(columns)

        # print("losses", self.kloss.item(), self.lloss.item(), self.sloss.item())
        #console.rule("end of train step")
        return self.return_tuple(self.sloss, self.output)

    @torch.no_grad()
    def valid_step(self, model_inputs, labels):
        # TODO: fix me! need to repair this to perform with eval!
        # self.model.set_layer_case("s")
        # self.run_preprocess(case="s")
        sret = self.model(model_inputs, case="s")
        ls = self.criterion(sret, labels)
        return self.return_tuple(ls, sret)
