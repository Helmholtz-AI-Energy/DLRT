from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg


class ProjectSVD:
    def __init__(self, network, max_vectors=None, min_s=1e-4):
        self.network = network
        self.max_vectors = max_vectors
        self.min_s = min_s
        self.param_buffers = {}
        for n, p in self.network.named_parameters():
            if not p.requires_grad or p.ndim == 1:
                continue

            # original plan was to average q then do the mult,
            # new plan is to average r, then do mult, then do average
            self.param_buffers[n] = {}
            self.param_buffers[n]["lpweights"] = None

    # @torch.no_grad()
    # def _project_vectors(
    #     self,
    #     basis0,
    #     basis1,
    #     s0,
    #     s1,
    #     max_vectors,
    #     rem_madnitude_difference: float = 1e-4,
    # ):
    #     # TODO: find a more efficient way of merging the vector spaces
    #     fact = {"dtype": basis0.dtype, "device": basis0.device}
    #     # ASSUMPTION: basis1 = a, basis0 == b,  target -> basis0
    #     # assume that they are both orthonormal bases
    #     # NOTE: both are assumed to be column-vectors (output of QR and SVD)
    #     # idea: decompose the vectors to the parallel and perp components
    #     # parallel component == ((a dot b) / (b dot b)) * b
    #     #       b dot b == 1 -> orthonormal
    #     # parallel component == (a dot b) * b
    #
    #     # TODO: determine if the values in the adjusted S should be along the diagonal, or should
    #     #   i represent some mixing? (this would be putting values on the off-diagonal)
    #     #       Lets start with the simple case of just doing the diagonal first
    #
    #     # 1. get parallel component with same order of vector (1 with 1, 2 with 2, etc)
    #     # 2. use these to scale the s values for the average???
    #     #   QUESTION: should the values be added to the target basis? it would make it not
    #     #   normalized...
    #     # 3. get the perpendicular components of the vectors from step 1
    #     # 4. get the parallel components of these vectors with the other vectors (1 with 2/3, etc)
    #     # 5. find how much of the remaining S value is for that vector
    #     # 6. if there is anything remaining in S, append the independent vector and append the
    #     #       rest of S to the end of the S0, sorting is unimportant!
    #     # 7. sum both S mats and divide by 2 (average)
    #
    #     # TODO: make sure that the vectors are linearly independent at the end
    #
    #     # step 1: get parallel components ------------------- ( matmul )
    #     parallel_mags = basis1.T @ basis0  # the rows of this matrix are the dot products
    #     # can use the diagonal elements to get the magnitude of the diagonal components
    #     comp_same_idx = torch.diag(parallel_mags)
    #     # parallel components are on the diagonal
    #     par_components = comp_same_idx * basis0
    #
    #     # 2. scale S values ---------------------------------
    #     # have all the scaling for the S values already (sum for parallel_mags)
    #     # reminder: s0 are the target s-vals, s1 is where we start because those are the
    #     #   magnitudes of basis1's vectors
    #     new_s1 = s1.clone() * torch.sum(parallel_mags, dim=0)
    #     # num S values should == num vecs in basis
    #
    #     # 3. get perpendicular comps ------------------------
    #     # can use the parallel_mags, but these magnitudes are of what is left!
    #     perp_components = basis1 - par_components  # this is mostly for the magnitudes
    #     # need to go through all the other vectors and grab everything above a certain level
    #     remaining_mags = torch.sqrt(1 - torch.pow(comp_same_idx, 2))
    #     # TODO: doulbe check that this is the same as `linalg.norm(perp_components, dim=0)`
    #     #   also, not sure which is more efficient...
    #
    #     # NEW IDEA: use householder transforms to make everything orthogonal
    #
    #     parallel_mags.fill_diagonal_(0)
    #     removed_cols = torch.zeros(s1.shape, device=s1.device, dtype=torch.bool)
    #
    #     # todo: only compare up to max_vectors!
    #     # print(remaining_mags)
    #     for cvec in range(basis0.shape[1]):
    #         # objective of this loop is to get the remaining vectors
    #         # print(cvec)
    #         # get parallel components, then perpendicular comps
    #         lp_par_comps = parallel_mags[:, cvec] * perp_components
    #         # print(perp_components)
    #         # print(lp_par_comps)
    #         perp_components = perp_components - lp_par_comps
    #         # need to get the new magnitudes of the remaining components:
    #         # print(cvec, remaining_mags, linalg.norm(lp_par_comps, dim=0))
    #         remaining_mags -= linalg.norm(lp_par_comps, dim=0)
    #         # TODO: should S be updated within the loop of outside of it?
    #         #   -> S is updated previously, this SHOULDN'T do anything new, it should just remove
    #         #       the parts of the perp_components which are pointing in the same direction
    #         #       but if there is no more magnitude remaining, it can be dropped
    #         removed_cols[remaining_mags <= rem_madnitude_difference] = True
    #         # remaining_mags[removed_cols] *= 0
    #         # set that aspect of perp_components to 0
    #         # perp_components[:, removed_cols] *= 0
    #         # if the remaining magnitude for anything hits 0, set the para
    #
    #     # # TODO: the squeeze might be an issue
    #     # removed_inds = torch.nonzero(removed_cols, as_tuple=False)
    #     # # print(remaining_mags, removed_cols, removed_inds)
    #     remaining_vectors = perp_components[:, ~removed_cols]
    #
    #     # if remaining_vectors.shape[1] > max_vectors:
    #     #     remaining_vectors = remaining_vectors[: max_vectors]
    #     # # Now, need to compare the perpendicular components to each other???? can happen during
    #     # #   the projection
    #     # # However, this time, the length of the vectors is not 1, need to do.....MATH
    #     #
    #     # # FIXME: something is still broken in here somewhere, im not sure what it is
    #     # #   the matrices should be linear independent but they are not.
    #     # #   maybe it has something to do with the abstraction to matrices?
    #     #
    #     # # need to get the magnitude of these
    #     # #   S-values should be their magnitude times the original S value
    #     # remaining_s_values = s1[~removed_cols] * linalg.norm(remaining_vectors, dim=0)
    #     # new_s1 = torch.cat([new_s1, remaining_s_values])
    #     # # average s matrices
    #     # new_s1[:s0.shape[0]] += s0  # the rest of the s0 values are 0
    #     # new_s1 /= 2.
    #
    #     # get the rest of perp_components which are non-zero
    #     # need to normalize the remaining vectors
    #     # remaining_vectors = F.normalize(remaining_vectors, dim=0)
    #     # join the outputs into a single basis (should be mostly linearly independent)
    #     out_basis = torch.cat([basis0, remaining_vectors], dim=1)
    #     return out_basis, new_s1

    @torch.no_grad()
    def _project_vectors_qr(
        self,
        u0,
        s0,
        vh0,
        u1,
        s1,
        vh1,
        max_vectors: int = -1,
        min_s: float = 1e-4,
    ):
        # merge the orthogonal vectors using QR then cutting off the remainder (if more than curoff)

        # cut off last vectors (max_vectors) and values
        v0, v1 = vh0.T, vh1.T
        if max_vectors == -1:
            max_vectors = int(u0.shape[1] * 1.5)

        # TODO: should the max vectors be here or below??
        # if max_vectors > 0:
        #     u0, u1 = u0[:, :max_vectors], u1[:, :max_vectors]
        #     s0, s1 = s0[:max_vectors], s1[:max_vectors]
        #     v0, v1 = v0[:, :max_vectors], v1[:, :max_vectors]

        # cat u's together on dim1
        ucat = torch.cat([u0, u1], dim=1)
        scat = torch.cat([s0, s1])
        vcat = torch.cat([v0, v1], dim=1)
        # do qr
        qu, ru = torch.linalg.qr(ucat, mode="reduced")
        qv, rv = torch.linalg.qr(vcat, mode="reduced")
        # multiply R by the cat of s
        # TODO: double check this math, should still be a vector, but still
        s = ru @ scat @ rv.T / 2.0
        # TODO: should this be abs or not?
        #   also, what hsould min_s be?
        to_keep = torch.abs(s) >= min_s
        s = s[to_keep]
        u = qu[:, to_keep]
        # sr = rv[to_keep, to_keep].T
        vh = qv[:, to_keep].T

        if max_vectors > 0:
            u = u[:, :max_vectors]
            s = s[:max_vectors]
            vh = vh[:max_vectors]
        return u, s, vh

    @staticmethod
    def _get_sizes(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor) -> torch.Tensor:
        sizes = torch.zeros(5, device=u.device, dtype=torch.int32)
        # factory = {"device": u.device, "dtype": u.dtype}
        sizes[0] = u.shape[0]
        sizes[1] = u.shape[1]
        sizes[2] = s.shape[0]
        sizes[3] = vh.shape[0]
        sizes[4] = vh.shape[1]
        # buffu = torch.empty(u.shape, **factory)
        return sizes

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
    def merge_svd(
        self,
        weights,
        max_vectors=-1,
        min_s: float = 1e-4,
    ) -> tuple[dist.Work, torch.Tensor]:
        # todo: make a tree merge pattern to merge all of the SVD stuff
        trans = False
        if weights.shape[0] < weights.shape[1]:
            weights = weights.T
            trans = True
        locu, locs, locvh = torch.linalg.svd(weights, full_matrices=False)
        sizes = self._get_sizes(locu, locs, locvh)
        factory = {"device": weights.device, "dtype": weights.dtype}
        buffu = torch.empty(sizes[:2], **factory)
        buffs = torch.empty(sizes[2], **factory)
        buffvh = torch.empty(sizes[3:], **factory)
        if max_vectors > 0:
            locu = locu[:, :max_vectors].contiguous()
            locs = locs[:max_vectors].contiguous()
            locvh = locvh[:max_vectors].contiguous()

            buffu = buffu[:, :max_vectors].contiguous()
            buffs = buffs[:max_vectors].contiguous()
            buffvh = buffvh[:max_vectors].contiguous()
        rank, size = dist.get_rank(), dist.get_world_size()
        # tree merge

        tree_width = size
        update_sizes = False
        while True:
            # todo: create new buffers each loop, the number of vecs may change...
            rem = int(tree_width % 2)
            cutoff = tree_width // 2
            # if rank >= cutoff:  # send
            # start at end of ranks (ws: 9 - start with 1-9 and leave rank 0)
            if rank < rem:
                tree_width -= cutoff
                # if tree_width <= 0:
                #     # should never happen in this loop, rems should always be touched
                #     break
                print(f"rank {rank} skipping, tree width: {tree_width}, rem: {rem}")
                continue
            if rank >= tree_width:
                # todo: make sure that this is okay
                print(f"rank {rank} breaking, tree width: {tree_width}, rem: {rem}")
                break
            target = rem + (rank % cutoff)  # should work on everything unless there is a remainder
            recv = rank <= cutoff
            send = rank > cutoff
            if send:
                print(f"rank {rank} merging, tree width: {tree_width}, rem: {rem}, pair: {target}")
                waitu = dist.isend(locu, dst=target, tag=0)
                waits = dist.isend(locs, dst=target, tag=1)
                waitvh = dist.isend(locvh, dst=target, tag=2)
                if update_sizes:
                    sizes = self._get_sizes(u=locu, s=locs, vh=locvh)
                    dist.send(sizes, dst=target, tag=3)
            else:  # if recv:
                print(
                    f"rank {rank} merging, tree width: {tree_width}, rem: {rem}, pair: {rank + cutoff}",
                )
                if update_sizes:
                    sizes.zero_()
                    dist.recv(sizes, src=rank + cutoff, tag=3)
                    buffu = torch.empty(sizes[:2], **factory)
                    buffs = torch.empty(sizes[2], **factory)
                    buffvh = torch.empty(sizes[3:], **factory)
                waitu = dist.irecv(buffu, src=rank + cutoff, tag=0)
                waits = dist.irecv(buffs, src=rank + cutoff, tag=1)
                waitvh = dist.irecv(buffvh, src=rank + cutoff, tag=2)

            waitu.wait()
            waits.wait()
            waitvh.wait()
            if send:
                # break off the other ranks which are not used anymore
                break

            # have the buffers on the recv ranks
            if recv:
                locu, locs, locvh = self._project_vectors(
                    u0=locu,
                    s0=locs,
                    vh0=locvh,
                    u1=buffu,
                    s1=buffs,
                    vh1=buffvh,
                    max_vectors=max_vectors,
                    min_s=min_s,
                )
                update_sizes = True

            tree_width -= cutoff
            if tree_width <= 1:
                break

        # TODO: remove me! DEBUG only
        dist.barrier()

        # TODO: reduce memory hit??
        new_weights = torch.linalg.multi_dot(locu, torch.diag(locs), locvh)
        weights = new_weights.T if trans else new_weights

        if rank != 0:
            weights *= 0
        # send the updated weights to all ranks (can be nonblocking)
        weights_wait = dist.broadcast(weights, src=0, async_op=True)

        return weights_wait, weights

    @torch.no_grad()
    def merge_weights(self, sync_level="all"):
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

            # make weights 2D in the case that they are not
            lpweights, shp = self._get_weight_shape(param=p)
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                lpweights = lpweights.T  # .contiguous())

            if self.param_buffers[n]["lpweights"] is None:
                self.param_buffers[n]["lpweights"] = lpweights.contiguous()
            else:
                self.param_buffers[n]["lpweights"].zero_()
                self.param_buffers[n]["lpweights"].add_(lpweights)

            wait2[n] = self.merge_svd(
                weights=self.param_buffers[n]["lpweights"],
                max_vectors=self.max_vectors,
                min_s=self.min_s,
            )
        for w in waits:
            w.wait()
        for key in wait2:
            wait2[key][0].wait()
        for n, p in self.network.named_parameters():
            if n in wait2:
                p.set_(wait2[n][1])


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
            self.param_buffers[n]["lpweights"] = None

        method_options = ["avg_q_avg_qr", "vecnorm", "wahba"]
        if method not in method_options:
            raise ValueError(f"method ({method}) must be one of {method_options}")
        self.project_method = method

    @torch.no_grad()
    def _iterate_param_buffers(self, skip_send=False):
        # TODO: abstract iterator??
        pass

    @torch.no_grad()
    def wahba_rotation(self, sync_level: str | None = None, qr_mode: str = "reduced") -> None:
        # merging based on the Wahba rotation matrix
        # 0: bcast the lpweights from rank0
        # 1: (2 ideas both use mean of dim=0)
        #   a: find the centroids of the points local to that process
        #   b: find the centroids of points on all processes
        # 2: find H -> H = (A - centroidsA) @ (B - centroidsB).T
        # 3: u, s, vh = svd(H)
        # 4: R = v.T @ u.T
        # 5: skipping translate step

        if sync_level is None:
            sync_level = "all"

        case = 1  # case for step 1

        waits = []
        # wait2 = {}
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
                continue
            if sync_level == "1d":
                continue

            lpweights, shp = self._get_weight_shape(param=p)
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                lpweights = lpweights.T  # .contiguous())

            if self.param_buffers[n]["lpweights"] is None:
                self.param_buffers[n]["lpweights"] = lpweights.contiguous()
            else:
                self.param_buffers[n]["lpweights"].zero_()
                self.param_buffers[n]["lpweights"].add_(lpweights)

            if self.rank != 0:
                self.param_buffers[n]["lpweights"].zero_()

            # step 0 - bcast lpweights (nonblocking)
            dist.broadcast(self.param_buffers[n]["lpweights"], src=0, async_op=False)
            # FIXME: skipping the overlap for now, can optimize after its working

            if self.rank != 0:  # don't need to do anything on rank 0 as it is not rotated
                # step1:
                if case == 1:
                    loc_mean = lpweights.mean(0)
                    rank0 = self.param_buffers[n]["lpweights"]
                    # h = (loc - mean_loc) @ (rank0 - mean_0).T
                    h = (lpweights - loc_mean) @ (rank0 - rank0.mean(0)).T
                    u, s, vh = torch.linalg.svd(h, full_matrices=True)
                    r = vh.T @ u.T
                    lpweights = r @ lpweights

            # new = lpweights.contiguous()
            # dist.all_reduce(new, op=dist.ReduceOp.AVG)
            # print(lpweights, shp, )
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                # NOTE: this will take more time!
                p.data.set_(lpweights.T.reshape(shp))  # .contiguous())
            else:
                p.data.set_(lpweights.reshape(shp))  # .contiguous())

            if sync_level == "all":
                p.data.set_(p.data.contiguous())
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))

        for w in waits:
            w.wait()

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
                dist.ReduceOp.AVG,
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
        elif self.project_method == "wahba":
            return self.wahba_rotation(sync_level=sync_level, qr_mode=qr_mode)


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
