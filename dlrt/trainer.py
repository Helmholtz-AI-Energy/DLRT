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
        scheduler=None,  # TODO: implement?
        rank_percent: float = None,
        adaptive: bool = True,
        ddp_dlrt_layers: bool = True,
        mixed_precision: bool = True,
        epsilon=None,
        split_batch: str = "repeat",
        dense_first_layer: bool = False,
        dense_last_layer: bool = False,
        pretrain_count: int = -1,
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
        self.criterion = criterion
        if split_batch not in ["repeat", "halfs", "thirds"]:
            raise ValueError("Unsupported option for split batch")
        self.split_batch = split_batch

        # replace linear layers
        self.torch_model = torch_model
        self.counter = 0
        self.pretrain_count = pretrain_count  # total number of pretraining steps

        self.dlrt_model = DLRTNetwork(
            torch_model=torch_model,
            rank_percent=rank_percent,
            adaptive=adaptive,
            epsilon=epsilon,
            ddp_dlrt_layers=ddp_dlrt_layers,
            dense_first_layer=dense_first_layer,
            dense_last_layer=dense_last_layer,
            pretrain_count=pretrain_count,
        )
        self.in_pretrain = lambda: self.counter < self.pretrain_count

        # need to rinit the optimizer with the new DLRT parameters
        optimizer_kwargs["params"] = self.dlrt_model.torch_model.parameters()
        self.optimizer = getattr(torch.optim, optimizer_name)(**optimizer_kwargs)
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            # to be used for printing only on the first rank
            rank = 0
        else:
            rank = 1
        if rank == 0:
            print(Pretty({"Optimizer": optimizer_name, **optimizer_kwargs}))

        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        if mixed_precision:
            # TODO: should there be different scalers for different parameters?
            #       i.e. -> one for k, onr for s, one for l
            self.kscaler = torch.cuda.amp.GradScaler()
            self.lscaler = torch.cuda.amp.GradScaler()
            self.sscaler = torch.cuda.amp.GradScaler()
            self.prescaler = torch.cuda.amp.GradScaler()

        self.return_tuple = namedtuple("Trainer", ["loss", "output"])

        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def _split_batch(self, inputs, labels):
        if self.split_batch == "repeat":
            # repeat the batch multiple times
            return (inputs, inputs, inputs), (labels, labels, labels)
        elif self.split_batch == "halfs":
            half = inputs.shape[0] // 2
            return (inputs[:half], inputs[half:], inputs), (labels[:half], labels[half:], labels)
        elif self.split_batch == "thirds":
            third1 = inputs.shape[0] // 3
            third2 = third1 * 2
            return (inputs[:third1], inputs[third1:third2], inputs[third2:]), (
                labels[:third1],
                labels[third1:third2],
                labels[third2:],
            )
        else:
            raise ValueError("Unsupported option for split batch")

    def _run_model(self, inputs, labels, case):
        # Steps:
        #   1. set layer case
        #   2. preprocess for that case (post process will be done in training step fn
        #   3. run forward/backward/opt step
        #   4. return
        self.dlrt_model.set_layer_case(case)
        self.dlrt_model.run_preprocess(case)
        self.optimizer.zero_grad()  # set_to_none=True)
        if self.mixed_precision:
            scaler = getattr(self, "kscaler")
            # scaler = getattr(self, f"{case}scaler")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.dlrt_model(inputs, case)
                loss = self.criterion(output, labels)
            scaler.scale(loss).backward()
            # nn.utils.clip_grad_norm_(self.dlrt_model.parameters(), max_norm=0.1)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            output = self.dlrt_model(inputs, case)
            loss = self.criterion(output, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(self.dlrt_model.parameters(), max_norm=0.1)
            self.optimizer.step()
        return loss, output

    def train_step_abs(self, inputs, labels):
        fact = {"device": inputs.device, "dtype": inputs.dtype}
        self.kloss, self.lloss, self.sloss = (
            torch.tensor(0, **fact),
            torch.tensor(0, **fact),
            torch.tensor(0, **fact),
        )
        # print(self.counter, self.pretrain_count, self.in_pretrain())
        if self.in_pretrain():
            self.dlrt_model.set_layer_case(case="pretrain")
            self.optimizer.zero_grad()  # set_to_none=True)
            if self.mixed_precision:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.dlrt_model(inputs, case="pretrain")
                    loss = self.criterion(output, labels)
                self.prescaler.scale(loss).backward()
                # nn.utils.clip_grad_norm_(self.dlrt_model.parameters(), max_norm=0.1)
                self.prescaler.step(self.optimizer)
                self.prescaler.update()
            else:
                output = self.dlrt_model(inputs, case="pretrain")
                loss = self.criterion(output, labels)
                loss.backward()
                # nn.utils.clip_grad_norm_(self.dlrt_model.parameters(), max_norm=0.1)
                self.optimizer.step()
            self.counter += 1
            if self.counter == self.pretrain_count:
                # convert the model here!
                print("stopping pretraining...")
                self.dlrt_model.stop_pretraining()
                # with torch.no_grad():
                #     self.dlrt_model.set_layer_case(case="k")
                #     afteroutputk = self.dlrt_model(inputs, case="k")
                #     afterlossk = self.criterion(output, labels)
                #     self.dlrt_model.set_layer_case(case="l")
                #     afteroutputl = self.dlrt_model(inputs, case="l")
                #     afterlossl = self.criterion(output, labels)
                # print(f"after pretraining k: {afterlossk}, l: {afterlossl}")

                # print(self.dlrt_model.dlrt_model)
            return (
                self.return_tuple(None, None),
                self.return_tuple(None, None),
                self.return_tuple(None, None),
                self.return_tuple(loss, output),
            )
        # -------------------------- DLRT --------------------------------
        split_inputs, split_labels = self._split_batch(inputs, labels)
        # TODO: splitting the batch into 3 sections...
        # ===== k step ====================
        kloss, koutput = self._run_model(split_inputs[0], split_labels[0], case="k")
        # kloss, koutput = self._run_model2(inputs, labels, case="k")
        # # ==== end k ==== start l ======
        # console.rule("l")
        lloss, loutput = self._run_model(split_inputs[1], split_labels[1], case="l")
        # lloss, loutput = self._run_model2(inputs, labels, case="l")

        # post process for both k and l
        self.dlrt_model.run_postprocess("k")
        self.dlrt_model.run_postprocess("l")
        # === end l === start s ===
        # sloss, soutput = self._run_model2(inputs, labels, case="s")
        sloss, soutput = self._run_model(split_inputs[2], split_labels[2], case="s")
        # rank adaptation ( + all reduce all DLRT params)
        if self.adaptive:
            self.dlrt_model.run_rank_adaption()

            if self.rank == 0 and self.counter % 10 == 0:
                console.rule(f"After rank adaptation - {self.counter}")
                columns = Columns(self.dlrt_model.get_all_ranks(), equal=True, expand=True)
                console.print(columns)
                console.rule()

        self.counter += 1
        if self.split_batch == "thirds":
            combiout = torch.cat([koutput, loutput, soutput], dim=0)
            combiloss = (kloss + lloss + sloss) / 3.0
        else:
            combiout = soutput
            combiloss = sloss
        return (
            self.return_tuple(kloss, koutput),
            self.return_tuple(lloss, loutput),
            self.return_tuple(sloss, soutput),
            self.return_tuple(combiloss, combiout),
        )

    @torch.no_grad()
    def valid_step(self, model_inputs, labels):
        # TODO: which stage should this be? k? l? s?
        # running the S-model -> this does not specify which case to use within the model!!
        # if self.in_pretrain():
        #     output = self.dlrt_model(model_inputs, None, True)
        #     loss = self.criterion(output, labels)
        #     return self.return_tuple(loss, output)
        if self.in_pretrain:
            sret = self.dlrt_model(model_inputs, "pretrain")
        else:
            sret = self.dlrt_model(model_inputs, "k")
        ls = self.criterion(sret, labels)
        return self.return_tuple(ls, sret)
