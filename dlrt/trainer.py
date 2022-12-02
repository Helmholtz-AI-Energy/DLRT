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
        self.reset_layers = None
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

        # requires_grad = []
        # for n, m in self.model.named_parameters():
        #    requires_grad.append(f"{n}, {m.requires_grad}")
        # if self.rank == 0:
        #    columns = Columns(requires_grad, equal=True, expand=True)
        #    console.rule("Original Model")
        #    console.print(columns)

        self.model = self.replace_linear_layers(self.model)
        # requires_grad = []
        # for n, m in self.model.named_parameters():
        #    requires_grad.append(f"{n}, {m.requires_grad}")
        # if self.rank == 0:
        #    columns = Columns(requires_grad, equal=True, expand=True)
        #    console.rule("DLRT Model")
        #    console.print(columns)

        # self.model = self.replace_linear_layers(self.model)
        # TODO: fix the last layer issue...
        # self.model = self._reset_last_layer_to_dense(self.model)
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
            # pass
            self.kmodel = self.model
            self.lmodel = self.model
            self.smodel = self.model

        self.rank = 0 if not dist.is_initialized() else dist.get_rank()
        # need to re-init the optimizer with the new DLRT parameters
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
        elif isinstance(module, nn.Conv1d):  # 2d
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
            # print(f"replacing {name} old: {module} with {module_output}")
            # del module
            # module = module_output

        for name, child in module.named_children():
            # print(name, child.extra_repr())
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
            m.training = totrain
            try:
                m.track_running_stats = totrain
            except AttributeError:
                pass

    def set_layer_case(self, case):
        models = [self.model]
        self.optimizer.zero_grad(set_to_none=True)
        if case in ["k", "l"]:
            # turn off training on all layers
            self._set_training_all_params(network=self.model, totrain=False)
            self.model.eval()
            try:
                self._set_training_all_params(network=self.kmodel, totrain=False)
                self._set_training_all_params(network=self.lmodel, totrain=False)
                models.append(getattr(self, f"{case}model"))
                self.kmodel.eval()
                self.lmodel.eval()
            except AttributeError:
                pass
        else:  # s case -> train all layers, turn off training of K and L
            self.model.train()
            self._set_training_all_params(network=self.model, totrain=True)
            try:
                self.smodel.train()
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

    def run_preprocess(self, case):
        # prev: getattr(self, f"{case}model")
        self.__run_command_on_dlrt_layers(module=self.model, command=f"{case}_preprocess")

    def run_postprocess(self, case):
        self.__run_command_on_dlrt_layers(module=self.model, command=f"{case}_postprocess")

    def run_rank_adaption(self):
        self.__run_command_on_dlrt_layers(module=self.model, command="rank_adaption")

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

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
        self.__collect_ranks(self.model)
        out_ranks = self.ranks.copy()
        self.ranks = []
        return out_ranks

    def _run_model(self, inputs, labels, case):
        self.optimizer.zero_grad(set_to_none=True)

        if self.kscaler is not None:  # only test for kscaler -> rest are not always defined
            scaler = getattr(self, f"{case}scaler")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = getattr(self, f"{case}model")(inputs)
                loss = self.criterion(output, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            # scaler.step(self.optimizer)
            # scaler.update()
        else:
            output = getattr(self, f"{case}model")(inputs)
            #output = self.model(inputs)
            #print(output)
            loss = self.criterion(output, labels)
            #print(loss)
            loss.backward()
            #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            # self.optimizer.step()
        #if torch.isnan(loss):
        #print("after step", case)
        #for n, m in getattr(self, f"{case}model").named_parameters():
        #    if n.startswith("fc"):
        #        print(n, f"{m.max().item():.4f}, {m.min().item():.4f}, {m.mean().item():.4f}, {m.std().item():.4f}")
        if torch.isnan(loss):
            raise RuntimeError(f"loss is NaN in case {case}! {output}")
        class_loss = getattr(self, f"{case}loss")
        class_loss *= 0.0
        class_loss += loss
        # TODO: remove output ???
        self.output = output
        return loss, output

    def train_step(self, inputs, labels, adapt=True):
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
            self.set_layer_case(case)
            self.run_preprocess(case)
            requires_grad = []
            #for n, m in self.kmodel.named_parameters():
            #    if n.startswith("fc"):  #m.requires_grad:
            #        requires_grad.append(f"{n}, {m.max().item():.4f}, {m.min().item():.4f}, {m.mean().item():.4f}, {m.std().item():.4f}, {m.requires_grad}")  # , {m.grad}, {m}")
            #if self.rank == 0: # and self.counter % 100 == 0:
            #    columns = Columns(requires_grad, equal=True, expand=True)
            #    console.rule(f"Before {case}")
            #    console.print(columns)
            self._run_model(inputs, labels, case)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.run_postprocess("k")
        self.run_postprocess("l")
        self.set_layer_case("s")
        self.run_preprocess(case="s")

        #requires_grad = []
        #for n, m in self.kmodel.named_parameters():
        #    if n.startswith("fc"):  #m.requires_grad:
        #        requires_grad.append(f"{n}, {m.max().item():.4f}, {m.min().item():.4f}, {m.mean().item():.4f}, {m.std().item():.4f}, {m.requires_grad}")  # , {m.grad}, {m}")
        #if self.rank == 0: # and self.counter % 100 == 0:
        #    columns = Columns(requires_grad, equal=True, expand=True)
        #    console.rule("Before s")
        #    console.print(columns)

        self._run_model(inputs, labels, "s")
        self.optimizer.step()
        if adapt and case == "s":
            if self.adaptive and adapt:
                self.run_rank_adaption()

                if self.rank == 0 and self.counter % 100 == 0:
                    columns = Columns(self.get_all_ranks(), equal=True, expand=True)
                    print(columns)
            self.counter += 1

        print("losses", self.kloss.item(), self.lloss.item(), self.sloss.item())
        return self.return_tuple(self.sloss, self.output)

    def train_step_old(self, model_inputs, labels, adapt=True):
        self.optimizer.zero_grad()
        # K
        self.set_layer_case("k")
        self.run_preproces(case="k")
        kloss, kout = self._run_model(model_inputs, labels, case="k")
        # requires_grad = []
        # for n, m in self.kmodel.named_parameters():
        #    if m.requires_grad:
        #        requires_grad.append(f"{n}, {m.requires_grad}")  # , {m.grad}, {m}")
        # if self.rank == 0 and self.counter % 100 == 0:
        #    columns = Columns(requires_grad, equal=True, expand=True)
        #    console.rule("k")
        #    console.print(columns)
        if torch.isnan(kloss):
            raise RuntimeError(f"kloss is NaN! {kout}")
        self.run_postproces(case="k")
        self.optimizer.zero_grad(set_to_none=True)

        # L
        # model_inputs = model_inputs.detach()
        self.set_layer_case("l")
        self.run_preproces(case="l")
        # requires_grad = []
        # for n, m in self.lmodel.named_parameters():
        #    requires_grad.append(f"{n}, {m.requires_grad}")
        # if self.rank == 0 and self.counter % 100 == 0:
        #    columns = Columns(requires_grad, equal=True, expand=True)
        #    console.rule("l")
        #    console.print(columns)
        lloss, _ = self._run_model(model_inputs, labels, case="l")
        self.optimizer.zero_grad(set_to_none=True)

        self.run_postproces(case="l")
        # end of no_sync??
        # model_inputs = model_inputs.detach()
        # S
        self.set_layer_case("s")
        self.run_preproces(case="s")
        # requires_grad = []
        # for n, m in self.lmodel.named_parameters():
        #    requires_grad.append(f"{n}, {m.requires_grad}")
        # if self.rank == 0 and self.counter % 100 == 0:
        #    columns = Columns(requires_grad, equal=True, expand=True)
        #    console.rule("s")
        #    console.print(columns)
        sloss, sret = self._run_model(model_inputs, labels, case="s")
        self.optimizer.step()

        # todo: set up scheduler
        if self.adaptive and adapt:
            self.run_rank_adaption()

            if self.rank == 0 and self.counter % 100 == 0:
                columns = Columns(self.get_all_ranks(), equal=True, expand=True)
                print(columns)
        self.counter += 1
        print("losses", kloss.item(), lloss.item(), sloss.item())
        return self.return_tuple(sloss, sret)

    @torch.no_grad()
    def valid_step(self, model_inputs, labels):
        # TODO: fix me! need to repair this to perform with eval!
        self.set_layer_case("s")
        #self.run_preprocess(case="s")
        sret = self.model(model_inputs)
        ls = self.criterion(sret, labels)
        return self.return_tuple(ls, sret)
