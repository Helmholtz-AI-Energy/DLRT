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

console = Console(width=140)

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
            dense_last_layer=True,
            init_ddp=init_ddp,
        )

        # need to rinit the optimizer with the new DLRT parameters
        optimizer_kwargs["params"] = self.model.model.parameters()
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
        # self.optimizer.zero_grad(set_to_none=True)

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

    def _run_model2(self, inputs, labels, case):
        # Steps:
        #   1. set layer case
        #   2. preprocess for that case (post process will be done in training step fn
        #   3. run forward/backward/opt step
        #   4. return
        self.model.set_layer_case(case)
        self.model.run_preprocess(case)
        self.optimizer.zero_grad()  # set_to_none=True)
        if self.mixed_precision:
            scaler = getattr(self, f"kscaler")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(inputs, case)
                loss = self.criterion(output, labels)
            scaler.scale(loss).backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            output = self.model(inputs, case)
            loss = self.criterion(output, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()
        return loss, output

    def train_step_abs(self, inputs, labels):
        fact = {"device": inputs.device, "dtype": inputs.dtype}
        self.kloss, self.lloss, self.sloss = (
            torch.tensor(0, **fact),
            torch.tensor(0, **fact),
            torch.tensor(0, **fact),
        )
        self.output = None
        # TODO: splitting the batch into 3 sections...
        half = inputs.shape[0] // 2
        # third2 = third * 2
        # ===== k step ====================
        kloss, koutput = self._run_model2(inputs[:half], labels[:half], case="k")
        # kloss, koutput = self._run_model2(inputs, labels, case="k")
        # # ==== end k ==== start l ======
        # console.rule("l")
        lloss, loutput = self._run_model2(inputs[half:], labels[half:], case="l")
        # lloss, loutput = self._run_model2(inputs, labels, case="l")

        # post process for both k and l
        self.model.run_postprocess("k")
        self.model.run_postprocess("l")
        # === end l === start s ===
        # sloss, soutput = self._run_model2(inputs, labels, case="s")
        sloss, soutput = self._run_model2(inputs, labels, case="s")
        # rank adaptation
        if self.adaptive:
            self.model.run_rank_adaption()

            if self.rank == 0 and self.counter % 10 == 0:
                console.rule(f"After rank adaptation - {self.counter}")
                columns = Columns(self.model.get_all_ranks(), equal=True, expand=True)
                console.print(columns)
                console.rule()

        self.counter += 1
        combiout = None  # torch.cat([koutput, loutput, soutput], dim=0)
        combiloss = (kloss + lloss + sloss) / 3.
        return self.return_tuple(kloss, koutput), self.return_tuple(
            lloss, loutput
        ), self.return_tuple(sloss, soutput), self.return_tuple(combiloss, combiout)

    @torch.no_grad()
    def valid_step(self, model_inputs, labels):
        # TODO: which stage should this be? k? l? s?
        sret = self.model(model_inputs, 's')
        ls = self.criterion(sret, labels)
        return self.return_tuple(ls, sret)
