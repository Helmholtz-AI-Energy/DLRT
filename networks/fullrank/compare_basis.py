from __future__ import annotations

from pathlib import Path
from queue import Queue

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn


@torch.no_grad()
class CompareQR:
    def __init__(
        self,
        network: torch.nn.Module,
        start_with_first_epoch: bool = True,
        mode="first",
    ):
        # this class is to be used to compare the Q values of each layer to the first epochs values
        # it will use pandas to aggregate the data
        # need to save 4 different dicts
        #   w   /  q
        #   w   /  q.T
        #   w.T /  q
        #   w.T /  q.T

        self.wq_dict = {}
        self.wqt_dict = {}
        self.wtq_dict = {}
        self.wtqt_dict = {}
        self.param_shapes = {}
        # self.baseline_weights = {}
        self.prevwq = Queue()
        self.prevwtq = Queue()

        self.mode = mode
        if mode == "first":
            self.nprev = 1
        elif mode.endswith("previous"):
            self.last_save = None
            if mode == "previous":
                self.nprev = 1
            else:
                # print(mode[:-8], mode)
                self.nprev = int(mode[:-8])
            self.holdw, self.holdwt = Queue(), Queue()
        else:
            raise ValueError(f"incorrect mode: {mode}. must be one of previous, first")

        for n, p in network.named_parameters():
            if p.ndim == 1:
                continue
            self.wq_dict[n] = {}
            self.wqt_dict[n] = {}
            self.wtq_dict[n] = {}
            self.wtqt_dict[n] = {}
            # self.baseline_weights[n] = {}
            self.param_shapes[n] = tuple(p.shape)

        # print(self.param_shapes)
        self.first_epoch = start_with_first_epoch

    def update_qrs(self, network, epoch, cdist=False, verbose=False):
        if dist.is_initialized() and dist.get_rank() > 0:
            return

        if self.nprev > epoch:  # put the first N epoch's weights into a queue
            # (will be retrieved in the same order)
            for n, p in network.named_parameters():
                if p.ndim == 1:
                    continue
                if p.ndim > 2:
                    lp = p.view(p.shape[0], -1)
                else:
                    lp = p
                w, _ = torch.linalg.qr(lp, mode="complete")
                wt, _ = torch.linalg.qr(lp.T, mode="complete")
                self.prevwq.put(w)
                self.prevwtq.put(wt)
            return

        for n, p in network.named_parameters():
            if p.ndim == 1:
                continue
            if p.ndim > 2:
                lp = p.view(p.shape[0], -1)
            else:
                lp = p
            # 4 cases: wq, wqt, wtq, wtqt
            # get the q values which are on top of the
            wq0 = self.prevwq.get()
            wqt0 = wq0.T
            wtq0 = self.prevwtq.get()
            wtqt0 = wtq0.T
            # wq + wqt
            wq, _ = torch.linalg.qr(lp, mode="complete")
            wq_perc = self._compare_q_cos(wq0, wq)
            wqt_perc = self._compare_q_cos(wqt0, wq.T)

            # wtq + wtqt
            wtq, _ = torch.linalg.qr(lp.T, mode="complete")
            wtq_perc = self._compare_q_cos(wtq0, wtq)
            wtqt_perc = self._compare_q_cos(wtqt0, wtq.T)

            if cdist:
                if lp.shape[0] > lp.shape[1]:  # out_ch > in_ch
                    mu, std, mn, mx, num_more_std = self._compare_cdist_diag(wq, wq0)
                else:
                    mu, std, mn, mx, num_more_std = self._compare_cdist_diag(wtq, wtq0)
                print(f"{n}, {mu:.4f}, {std:.4f}, {mn:.4f}, {mx:.4f}, {num_more_std:.4f}")

            self.wq_dict[n][epoch] = wq_perc
            self.wqt_dict[n][epoch] = wqt_perc
            self.wtq_dict[n][epoch] = wtq_perc
            self.wtqt_dict[n][epoch] = wtqt_perc
            if verbose:
                if lp.shape[0] > lp.shape[1]:  # more rows than columns, i.e. out_ch > in_ch
                    print(f"{n}\twq: {wq_perc:.4f}, wqt: {wqt_perc:.4f}\t{lp.shape}")
                else:
                    print(f"{n}\twtq: {wtq_perc:.4f}, wtqt: {wtqt_perc:.4f}\t{lp.shape}")
            if self.mode != "first":
                self.prevwq.put(wq)
                self.prevwtq.put(wtq)
            else:
                self.prevwq.put(wq0)
                self.prevwtq.put(wtq0)
        if not verbose:
            print("Finished with q comp")

    @staticmethod
    def _compare_q_cos(q1, q2, k=1):
        # use the cosine distance to determine which angle is most similar
        # return the fraction of
        cos_ang = q1 @ q2.T
        arange = torch.arange(cos_ang.shape[0], device=cos_ang.device)
        top_angles = torch.topk(cos_ang, k=k, largest=True)[1]
        if k > 1:
            in_top = torch.tensor(
                [a in b for a, b in zip(top_angles, arange)],
                dtype=torch.bool,
                device=top_angles.device,
            )
            return (in_top.sum() / top_angles.shape[0]).item()
        sum_same = (top_angles.flatten() == arange).sum()
        return (sum_same / top_angles.shape[0]).item()
        # print(
        #     epoch, f"\tnumber in the top{k} angles:", in_top3.sum().item(),
        #     f"\tnumber of elements:", top_angles.shape[0], "\t%:",
        #     f"{((in_top3.sum() / top_angles.shape[0]) * 100).item():.4f}"
        # )

    @staticmethod
    def _compare_cdist_diag(q1, q2):  # return average change of vector
        # since Q is assumed to consist of column vectors, need to transpose them
        cdist = torch.diag(torch.cdist(q1.T, q2.T, p=2))
        mu, std = cdist.mean().item(), cdist.std().item()
        mn, mx = cdist.min().item(), cdist.max().item()
        more_than_one_std = (cdist >= (mu + std)).sum().item() / cdist.shape[0]
        return mu, std, mn, mx, more_than_one_std

    def generate_pd_dfs(self, save=True, out_folder=None):
        self.wq_df = pd.DataFrame.from_dict(self.wq_dict)
        self.wtq_df = pd.DataFrame.from_dict(self.wtq_dict)
        self.wqt_df = pd.DataFrame.from_dict(self.wqt_dict)
        self.wtqt_df = pd.DataFrame.from_dict(self.wtqt_dict)
        if save and out_folder is None:
            raise ValueError("out_folder must be specified to save qr comparisons")
        if save:
            self.save_dfs(out_folder)

    def save_dfs(self, folder):
        out_folder = Path(folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        self.wq_df.to_csv(out_folder / f"wq-{self.mode}.csv", index=False)
        self.wtq_df.to_csv(out_folder / f"wtq-{self.mode}.csv", index=False)
        self.wqt_df.to_csv(out_folder / f"wqt-{self.mode}.csv", index=False)
        self.wtqt_df.to_csv(out_folder / f"wtqt-{self.mode}.csv", index=False)


@torch.no_grad()
class QRProjectWeights:
    def __init__(self, delay, skip_first_layer=True):
        self.delay = delay
        self.skip_first_layer = skip_first_layer
        self.tracking_q = Queue()

    @torch.no_grad()
    def project_weights(self, network, epoch):
        self._avg_q_avg_r(network)
        # if epoch < self.delay:
        #     return self._set_qtracking(network)

        # print(epoch, self.delay)
        # if epoch == self.delay - 2 or self.delay == 1:

        # project the weights to the original Q basis, then sync, then update tracking q
        # self._project_q(network)

    @torch.no_grad()
    def _set_qtracking(self, network):
        first = self.skip_first_layer
        for n, p in network.named_parameters():
            if p.ndim == 1:
                continue
            if first:
                first = False
                continue
            if p.ndim > 2:
                lp = p.view(p.shape[0], -1)
            else:
                lp = p

            if lp.shape[0] > lp.shape[1]:
                q, _ = torch.linalg.qr(lp, mode="complete")
            else:
                q, _ = torch.linalg.qr(lp.T, mode="complete")
            self.tracking_q.put(q)

    @torch.no_grad()
    def _project_q(self, network):
        # DOESNT WORK -> doesnt converge at all :(
        first = self.skip_first_layer
        for n, p in network.named_parameters():
            if p.ndim == 1:
                continue
            if first:
                first = False
                dist.all_reduce(p, dist.ReduceOp.AVG)
                continue
            shp = p.shape
            if p.ndim > 2:
                lp = p.view(p.shape[0], -1)
            else:
                lp = p

            qbase = self.tracking_q.get()
            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lp.shape[0] > lp.shape[1]:
                # TODO: test mode=r / reduced
                localq, localr = torch.linalg.qr(lp, mode="complete")
                new = qbase @ localr
                p.set_(new.view(shp).contiguous())
            else:
                localq, localr = torch.linalg.qr(lp.T, mode="complete")
                new = qbase @ localr
                p.set_(new.T.view(shp).contiguous())

            if dist.is_initialized():
                dist.all_reduce(p, dist.ReduceOp.AVG)

            if lp.shape[0] > lp.shape[1]:
                qbase, _ = torch.linalg.qr(lp, mode="complete")
            else:
                qbase, _ = torch.linalg.qr(lp.T, mode="complete")

            self.tracking_q.put(qbase)

    @staticmethod
    def _avg_q_avg_r(network):
        print("custom reduction")
        for n, p in network.named_parameters():
            if p.ndim == 1:
                continue
            shp = p.shape
            if p.ndim > 2:
                lp = p.view(p.shape[0], -1)
            else:
                lp = p

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            localq, localr = torch.linalg.qr(lp, mode="reduced")
            localq = localq.contiguous()
            dist.all_reduce(localq, dist.ReduceOp.AVG)
            new = localq @ localr
            p.set_(new.view(shp).contiguous())

            if dist.is_initialized():
                dist.all_reduce(p, dist.ReduceOp.AVG)


class ProjectGrads:
    def __init__(self, network: nn.Module):
        self.shapes = {}
        self.nelems = {}
        self.slices = {}
        st = 0
        for n, p in network.named_parameters():
            self.shapes[n] = tuple(p.shape)
            self.nelems[n] = p.nelem
            self.slices[n] = slice(st, p.nelem + st)
            st += p.nelem
        self.network = network

    def set_hook(self):
        # instead of working at the comm hook level, we will work at the backward hook level

        pass
