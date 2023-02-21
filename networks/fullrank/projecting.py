from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ProjectSVD:
    def __init__(self, network):
        self.network = network


class ProjectWeightsHoldQ:
    def __init__(self, network):
        self.network = network
        self.rank = dist.get_rank()
        self.param_buffers = {}
        self.hold_q = {}
        self.hold_q_grads = {}

        self.first_update = 0
        self.last_update = None
        self.smoothing = 2
        self.period = 10 + 1

    @torch.no_grad()
    def project_r(self):
        if len(self.hold_q.keys()) == 0:  # if there are no bases to work from
            return self.update_and_project()
        # This one is only when B is known
        waits = []
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
                continue

            weights = p.data
            shp = weights.shape

            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                _, localr = torch.linalg.qr(lpweights, mode="complete")
                trans = False
            else:
                _, localr = torch.linalg.qr(lpweights.T, mode="complete")
                trans = True

            # TODO: should R be averaged??
            localr = localr.contiguous()
            dist.all_reduce(localr, dist.ReduceOp.AVG, async_op=False)

            new = self.hold_q[n] @ localr
            if trans:
                p.data.set_(new.T.view(shp))  # .contiguous())
            else:
                p.data.set_(new.view(shp))  # .contiguous())

        for w in waits:
            w.wait()

    @torch.no_grad()
    def update_and_project(self, skip_avg=False):
        waits = []
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
                continue

            weights = p.data
            shp = weights.shape

            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                localq, localr = torch.linalg.qr(lpweights, mode="complete")
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode="complete")
                trans = True
            localq = localq.contiguous()
            dist.all_reduce(localq, op=dist.ReduceOp.AVG)  # blocking...
            # TODO: average localq before doing this??
            self._update_held_q_ema(name=n, new_q=localq)

            new = self.hold_q[n] @ localr
            if trans:
                p.data.set_(new.T.view(shp))  # .contiguous())
            else:
                p.data.set_(new.view(shp))  # .contiguous())

            if not skip_avg:
                waits.append(dist.all_reduce(p.data, op=dist.ReduceOp.AVG, async_op=True))

        for w in waits:
            w.wait()

    @torch.no_grad()
    def _update_held_q_ema(self, name, new_q):
        if name not in self.hold_q:
            self.hold_q[name] = new_q
        smoothperiod = self.smoothing / self.period
        self.hold_q[name] = new_q * smoothperiod + self.hold_q[name] * (1 - smoothperiod)

    @torch.no_grad()
    def update_and_project_grads(self, skip_avg=False):
        waits = []
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.grad, dist.ReduceOp.AVG, async_op=True))
                continue

            grads = p.grad
            shp = grads.shape

            if grads.ndim > 2:
                lpgrads = grads.view(grads.shape[0], -1)
            else:
                lpgrads = grads

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpgrads.shape[0] >= lpgrads.shape[1]:  # already TS of similar
                localq, localr = torch.linalg.qr(lpgrads, mode="reduced")
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpgrads.T, mode="reduced")
                trans = True
            localq = localq.contiguous()
            dist.all_reduce(localq, op=dist.ReduceOp.AVG)  # blocking...
            # TODO: average localq before doing this??
            self._update_held_q_ema_grads(name=n, new_q=localq)

            new = self.hold_q_grads[n] @ localr
            if trans:
                p.grad.set_(new.T.view(shp).contiguous())
            else:
                p.grad.set_(new.view(shp).contiguous())

            if not skip_avg:
                waits.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True))

        for w in waits:
            w.wait()

    @torch.no_grad()
    def _update_held_q_ema_grads(self, name, new_q):
        if name not in self.hold_q_grads:
            self.hold_q_grads[name] = new_q
        smoothperiod = self.smoothing / self.period
        self.hold_q_grads[name] = new_q * smoothperiod + self.hold_q_grads[name] * (1 - smoothperiod)

    @torch.no_grad()
    def project_weights_old(self, force=False, only1d=False):
        # todo: linear, conv2d, batchnorm, more?
        # fwr = u @ s @ v.T  # full weight representation
        # conv weights -> out_ch, in_ch / groups, *kern size
        #   merge the kernels for each channel?
        #   test both
        # bias - 1D (output shape)
        #   make this into a diagonal, then do the projection?
        #   can do 1/bias for the inverse
        # if weights.ndim == 1:
        #     average weights instead???
        # linear - 2D weights

        # self.start_qsend()
        # self.update_after_send()

        if force:
            self.sync_level = 2
            # end_sync = True
            only1d = False
        elif only1d:
            pass
        else:
            return

        waits = []
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
                continue
            if only1d:
                continue

            weights = p.data
            shp = weights.shape

            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                localq, localr = torch.linalg.qr(lpweights, mode="reduced")
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode="reduced")
                trans = True

            # if self.sync_level == 1:  # only sync Q (TODO: maybe only sync R??
            # localr = localr.contiguous()
            # dist.all_reduce(localr, dist.ReduceOp.AVG, async_op=False)
            #     # dist.all_reduce(localq, dist.ReduceOp.AVG)
            #     new = localq @ localr  # ) / dist.get_world_size()
            #     if trans:
            #         p.data.set_(new.T.view(shp))  # .contiguous())
            #     else:
            #         p.data.set_(new.view(shp))  # .contiguous())
            # if self.sync_level == 2:
            localq = localq.contiguous()
            # dist.all_reduce(localr, dist.ReduceOp.AVG, async_op=True)
            dist.all_reduce(localq, dist.ReduceOp.AVG)
            new = localq @ localr  # ) / dist.get_world_size()
            if trans:
                p.data.set_(new.T.view(shp).contiguous())
            else:
                p.data.set_(new.view(shp).contiguous())

            # if self.sync_level == 2:
            p.data.set_(p.data.contiguous())
            waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
            # self.tosync = False
            # dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=False)
        # if self.rank == 0:
        #     print("before waits")
        for w in waits:
            w.wait()

    @torch.no_grad()
    def project_weights(self, sync_level="all"):
        # todo: linear, conv2d, batchnorm, more?
        # fwr = u @ s @ v.T  # full weight representation
        # conv weights -> out_ch, in_ch / groups, *kern size
        #   merge the kernels for each channel?
        #   test both
        # bias - 1D (output shape)
        #   make this into a diagonal, then do the projection?
        #   can do 1/bias for the inverse
        # if weights.ndim == 1:
        #     average weights instead???
        # linear - 2D weights

        # sync_levels:
        #   '1d': only 1d
        #   'r': only r values
        #   'q': only q values
        #   'all': r values, then weights

        waits = []
        wait2 = {}
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
                continue
            if sync_level == "1d":
                continue

            weights = p.data
            shp = weights.shape

            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                localq, localr = torch.linalg.qr(lpweights, mode="reduced")
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode="reduced")
                trans = True
            if sync_level in ["r", "all"]:  # sync only R
                self.param_buffers[n]["r"].zero_()
                self.param_buffers[n]["r"].add_(localr)  # need contiguous tensor
                self.param_buffers[n]["q"].zero_()
                self.param_buffers[n]["q"].add_(localq)
                wait2[n] = dist.all_reduce(
                    self.param_buffers[n]["r"],
                    dist.ReduceOp.AVG,
                    async_op=True,
                )
            elif sync_level == "q":
                self.param_buffers[n]["r"].zero_()
                self.param_buffers[n]["r"].add_(localr)  # need contiguous tensor
                self.param_buffers[n]["q"].zero_()
                self.param_buffers[n]["q"].add_(localq)
                wait2[n] = dist.all_reduce(
                    self.param_buffers[n]["q"],
                    dist.ReduceOp.AVG,
                    async_op=True,
                )

        for n, p in self.network.named_parameters():
            if not p.requires_grad or p.ndim == 1 or sync_level == "1d":
                continue

            weights = p.data
            shp = weights.shape

            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                trans = False
            else:
                trans = True
            wait2[n].wait()
            new = self.param_buffers[n]["q"] @ self.param_buffers[n]["r"]
            if trans:
                p.data.set_(new.T.view(shp))  # .contiguous())
            else:
                p.data.set_(new.view(shp))  # .contiguous())

            if sync_level == "all":
                # p.data.set_(p.data.contiguous())
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
        for w in waits:
            w.wait()


class ProjectWeightsQR:
    def __init__(self, network: nn.Module, method: str = "vecnorm"):
        self.network = network
        self.rank = dist.get_rank()
        self.param_buffers = {}
        for n, p in self.network.named_parameters():
            if not p.requires_grad or p.ndim == 1:
                continue

            # original plan was to average q then do the mult,
            # new plan is to average r, then do mult, then do average
            self.param_buffers[n] = {}
            self.param_buffers[n]["q"] = None
            self.param_buffers[n]["r"] = None

        method_options = ["avg_q_avg_qr", "vecnorm", "project_rank0"]
        if method not in method_options:
            raise ValueError(f"method ({method}) must be one of {method_options}")
        self.project_method = method

    @torch.no_grad()
    def _iterate_param_buffers(self, skip_send=False):
        # TODO: abstract iterator??
        pass

    @torch.no_grad()
    def _set_qr_dict(self, name, q, r):
        if self.param_buffers[name]["r"] is None:
            self.param_buffers[name]["r"] = r.contiguous()
        else:
            self.param_buffers[name]["r"].zero_()
            self.param_buffers[name]["r"].add_(r)  # need contiguous tensor

        if self.param_buffers[name]["q"] is None:
            self.param_buffers[name]["q"] = q.contiguous()
        else:
            self.param_buffers[name]["q"].zero_()
            self.param_buffers[name]["q"].add_(q)

    @torch.no_grad()
    def _get_weight_shape(self, param):
        weights = param.data
        shp = weights.shape

        if weights.ndim > 2:
            lpweights = weights.view(weights.shape[0], -1)
        else:
            lpweights = weights
        return lpweights, shp

    @torch.no_grad()
    def sum_vecnorm_merge(self, sync_level: str | None = None, qr_mode: str = "reduced") -> None:
        """
        This will do a sum of the Q matrices, then it will normalize them to make Q orthogonal
        once more. Then it will find the weights W from the new QR, then it averages the weights

        Attributes
        ----------
        sync_level: optional, str
            control what should be synced. options:
                all: (default) do full sync
                q: only sync q as detailed, skip full w sync
        qr_mode: str
            mode for the QR decomp. default: "reduced"
        """
        if sync_level is None:
            sync_level = "all"
        # TODO: should the BN be viewed as an orthogonal transform?
        # in this method, sum up the Q matrices, then normalize them all to be orthonormal again
        waits = []
        wait2 = {}
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
                continue

            lpweights, _ = self._get_weight_shape(param=p)

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                localq, localr = torch.linalg.qr(lpweights, mode=qr_mode)
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode=qr_mode)
            # if sync_level in ['r', 'all']:  # sync only R
            self._set_qr_dict(name=n, q=localq, r=localr)

            wait2[n] = dist.all_reduce(
                self.param_buffers[n]["q"],
                dist.ReduceOp.SUM,
                async_op=True,
            )

        for n, p in self.network.named_parameters():
            if not p.requires_grad or p.ndim == 1:
                continue

            lpweights, shp = self._get_weight_shape(param=p)

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                trans = False
            else:
                trans = True
            wait2[n].wait()
            # =========== difference from other methods =================
            q = self.param_buffers[n]["q"]
            self.param_buffers[n]["q"] = F.normalize(q, dim=1, p=2.0)
            # =========== difference from other methods =================

            new = self.param_buffers[n]["q"] @ self.param_buffers[n]["r"]
            if trans:
                p.data.set_(new.T.view(shp))  # .contiguous())
            else:
                p.data.set_(new.view(shp))  # .contiguous())
            if sync_level == "all":
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
        for w in waits:
            w.wait()

    @torch.no_grad()
    def avg_q_avg_qr(self, sync_level: str | None, qr_mode: str = "reduced") -> None:
        """
        Average the Q mats, then calc QR to get the new weight repr, then average the new weights

        1D weights are averaged in the normal way
        TODO: they should be updated every step with normal gradients (DDP style)

        Parameters
        ----------
        sync_level: str, optional
            What should be synced:
                '1d': only 1d
                'r': only r values
                'q': only q values
                'all': q values, then weights
        qr_mode: str
            mode for the qr decomp, default: "reduced"
        """
        if sync_level is None:
            sync_level = "all"

        waits = []
        wait2 = {}
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
                continue
            if sync_level == "1d":
                continue

            lpweights, _ = self._get_weight_shape(param=p)

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                localq, localr = torch.linalg.qr(lpweights, mode=qr_mode)
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode=qr_mode)
            self._set_qr_dict(name=n, q=localq, r=localr)

            # if sync_level in ['r', 'all']:  # sync only R
            if sync_level == "r":  # sync only R
                wait2[n] = dist.all_reduce(
                    self.param_buffers[n]["r"],
                    dist.ReduceOp.AVG,
                    async_op=True,
                )
            elif sync_level in ["q", "all"]:
                # elif sync_level == 'q':
                wait2[n] = dist.all_reduce(
                    self.param_buffers[n]["q"],
                    dist.ReduceOp.AVG,
                    async_op=True,
                )

        for n, p in self.network.named_parameters():
            if not p.requires_grad or p.ndim == 1 or sync_level == "1d":
                continue

            lpweights, shp = self._get_weight_shape(param=p)

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                trans = False
            else:
                trans = True
            wait2[n].wait()

            new = self.param_buffers[n]["q"] @ self.param_buffers[n]["r"]
            if trans:
                p.data.set_(new.T.view(shp))  # .contiguous())
            else:
                p.data.set_(new.view(shp))  # .contiguous())

            if sync_level == "all":
                # p.data.set_(p.data.contiguous())
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
        for w in waits:
            w.wait()

    @torch.no_grad()
    def project_weights_old(self, force=False, only1d=False):
        if force:
            self.sync_level = 2
            # end_sync = True
            only1d = False
        elif only1d:
            pass
        else:
            return

        waits = []
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
                continue
            if only1d:
                continue

            weights = p.data
            shp = weights.shape

            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                localq, localr = torch.linalg.qr(lpweights, mode="reduced")
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode="reduced")
                trans = True

            # if self.sync_level == 1:  # only sync Q (TODO: maybe only sync R??
            # localr = localr.contiguous()
            # dist.all_reduce(localr, dist.ReduceOp.AVG, async_op=False)
            #     # dist.all_reduce(localq, dist.ReduceOp.AVG)
            #     new = localq @ localr  # ) / dist.get_world_size()
            #     if trans:
            #         p.data.set_(new.T.view(shp))  # .contiguous())
            #     else:
            #         p.data.set_(new.view(shp))  # .contiguous())
            # if self.sync_level == 2:
            localq = localq.contiguous()
            # dist.all_reduce(localr, dist.ReduceOp.AVG, async_op=True)
            dist.all_reduce(localq, dist.ReduceOp.AVG)
            new = localq @ localr  # ) / dist.get_world_size()
            if trans:
                p.data.set_(new.T.view(shp).contiguous())
            else:
                p.data.set_(new.view(shp).contiguous())

            # if self.sync_level == 2:
            p.data.set_(p.data.contiguous())
            waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
            # self.tosync = False
            # dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=False)
        # if self.rank == 0:
        #     print("before waits")
        for w in waits:
            w.wait()

    @torch.no_grad()
    def project_weights(self, sync_level="all", qr_mode="reduced"):  # noqa: C901
        if self.project_method == "vecnorm":
            return self.sum_vecnorm_merge(sync_level=sync_level, qr_mode=qr_mode)
        elif self.project_method == "avg_q_avg_qr":
            return self.avg_q_avg_qr(sync_level=sync_level, qr_mode=qr_mode)


@torch.no_grad()
def project_weights(network: nn.Module):

    # rank = dist.get_rank()
    # todo: linear, conv2d, batchnorm, more?
    # fwr = u @ s @ v.T  # full weight representation
    # conv weights -> out_ch, in_ch / groups, *kern size
    #   merge the kernels for each channel?
    #   test both
    # bias - 1D (output shape)
    #   make this into a diagonal, then do the projection?
    #   can do 1/bias for the inverse
    # if weights.ndim == 1:
    #     average weights instead???
    # linear - 2D weights
    for n, p in network.named_parameters():
        if not p.requires_grad:
            continue
        # p.grad = F.normalize(p.grad, p=2.0, dim=1 if p.grad.ndim > 1 else 0)
        if p.ndim == 1:
            dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True)
            continue

        weights = p.data
        # compare = grads.clone()
        # dist.all_reduce(compare, dist.ReduceOp.AVG)

        shp = weights.shape

        if weights.ndim > 2:
            lpweights = weights.view(weights.shape[0], -1)
        else:
            lpweights = weights

        # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
        if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
            localq, localr = torch.linalg.qr(lpweights, mode="reduced")
            trans = False
        else:
            localq, localr = torch.linalg.qr(lpweights.T, mode="reduced")
            trans = True
        localq = localq.contiguous()
        # localr = localr.contiguous()
        # dist.all_reduce(localr, dist.ReduceOp.AVG, async_op=True)
        dist.all_reduce(localq, dist.ReduceOp.AVG)
        new = localq @ localr  # ) / dist.get_world_size()
        if trans:
            p.data.set_(new.T.view(shp).contiguous())
        else:
            p.data.set_(new.view(shp).contiguous())

        # compare -= p.grad
        # print(f"{compare.mean().item():.5f}, {compare.std().item():.5f}, {compare.min().item():.5f}, "
        #       f"{compare.max().item():.5f}")

        dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True)


@torch.no_grad()
def project_grads_hook_old(network: nn.Module):

    # rank = dist.get_rank()
    # todo: linear, conv2d, batchnorm, more?
    # fwr = u @ s @ v.T  # full weight representation
    # conv weights -> out_ch, in_ch / groups, *kern size
    #   merge the kernels for each channel?
    #   test both
    # bias - 1D (output shape)
    #   make this into a diagonal, then do the projection?
    #   can do 1/bias for the inverse
    # if weights.ndim == 1:
    #     average weights instead???
    # linear - 2D weights
    for n, p in network.named_parameters():
        if not p.requires_grad:
            continue
        # p.grad = F.normalize(p.grad, p=2.0, dim=1 if p.grad.ndim > 1 else 0)
        if p.ndim == 1:
            dist.all_reduce(p.grad, dist.ReduceOp.AVG, async_op=True)
            continue

        grads = p.grad
        # compare = grads.clone()
        # dist.all_reduce(compare, dist.ReduceOp.AVG)

        shp = grads.shape

        if grads.ndim > 2:
            lpgrads = grads.view(grads.shape[0], -1)
        else:
            lpgrads = grads

        # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
        localq, localr = torch.linalg.qr(lpgrads.T, mode="complete")
        localq = localq.contiguous()
        # localr = localr.contiguous()
        # dist.all_reduce(localr, dist.ReduceOp.AVG, async_op=True)
        dist.all_reduce(localq, dist.ReduceOp.AVG)
        new = localq @ localr  # ) / dist.get_world_size()
        p.grad.set_(new.T.view(shp).contiguous())

        # compare -= p.grad
        # print(f"{compare.mean().item():.5f}, {compare.std().item():.5f}, {compare.min().item():.5f}, "
        #       f"{compare.max().item():.5f}")

        dist.all_reduce(p.grad, dist.ReduceOp.AVG)
        # normalize between -1 and 1 ???? no normalize???
        # p.grad = 2 * ((p.grad - p.grad.min()) / (p.grad.max() - p.grad.min())) - 1.
        # p.grad = F.normalize(p.grad, p=2.0, dim=None)  #1 if p.grad.ndim > 1 else 0)

    # grads /= dist.get_world_size()
    # dist.all_reduce(grads, op=dist.ReduceOp.AVG)
    # return new_grads
