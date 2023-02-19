import torch
import torch.nn as nn
import torch.distributed as dist


class ProjectSVD(object):
    def __init__(self, network):
        self.network = network


class ProjectWeightsHoldQ(object):
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
                _, localr = torch.linalg.qr(lpweights, mode='complete')
                trans = False
            else:
                _, localr = torch.linalg.qr(lpweights.T, mode='complete')
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
                localq, localr = torch.linalg.qr(lpweights, mode='complete')
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode='complete')
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
                localq, localr = torch.linalg.qr(lpgrads, mode='reduced')
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpgrads.T, mode='reduced')
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
                localq, localr = torch.linalg.qr(lpweights, mode='reduced')
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode='reduced')
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
    def project_weights(self, sync_level='all'):
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
                localq, localr = torch.linalg.qr(lpweights, mode='reduced')
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode='reduced')
                trans = True
            if sync_level in ['r', 'all']:  # sync only R
                self.param_buffers[n]['r'].zero_()
                self.param_buffers[n]['r'].add_(localr)  # need contiguous tensor
                self.param_buffers[n]['q'].zero_()
                self.param_buffers[n]['q'].add_(localq)
                wait2[n] = dist.all_reduce(
                    self.param_buffers[n]['r'], dist.ReduceOp.AVG, async_op=True
                )
            elif sync_level == 'q':
                self.param_buffers[n]['r'].zero_()
                self.param_buffers[n]['r'].add_(localr)  # need contiguous tensor
                self.param_buffers[n]['q'].zero_()
                self.param_buffers[n]['q'].add_(localq)
                wait2[n] = dist.all_reduce(
                    self.param_buffers[n]['q'], dist.ReduceOp.AVG, async_op=True
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
            new = self.param_buffers[n]['q'] @ self.param_buffers[n]['r']
            if trans:
                p.data.set_(new.T.view(shp))#.contiguous())
            else:
                p.data.set_(new.view(shp))#.contiguous())

            if sync_level == "all":
                # p.data.set_(p.data.contiguous())
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
        for w in waits:
            w.wait()


class ProjectWeightsQR(object):
    def __init__(self, network):
        self.network = network
        self.rank = dist.get_rank()
        self.param_buffers = {}
        for n, p in self.network.named_parameters():
            if not p.requires_grad or p.ndim == 1:
                continue

            weights = p.data
            shp = weights.shape

            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights

            # note: reduced QR will double the amount of data sent
            #   (reduced) Q.shape == lpweights.shape
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                q_shp = (lpweights.shape[0], lpweights.shape[1])
                r_shp = (lpweights.shape[1], lpweights.shape[1])
            else:
                q_shp = (lpweights.shape[1], lpweights.shape[0])
                r_shp = (lpweights.shape[0], lpweights.shape[0])

            # original plan was to average q then do the mult,
            # new plan is to average r, then do mult, then do average
            self.param_buffers[n] = {}
            factory = {"device": p.device, "dtype": p.dtype}
            self.param_buffers[n]['q'] = None  # torch.empty(q_shp, **factory)
            self.param_buffers[n]['r'] = None  # torch.empty(r_shp, **factory)

        # self.nsync_params = None
        # self.to_sync_wait = None
        # self.last_loss1 = None
        # self.last_loss2 = None
        # self.tosync = True
        # self.sync_level = 0

    @torch.no_grad()
    def test_ifsync(self, loss=None):
        # # tosync = []
        # losses = torch.zeros(dist.get_world_size(), device=loss.device, dtype=loss.dtype)
        # losses[self.rank] = loss
        # dist.all_reduce(losses, op=dist.ReduceOp.SUM, async_op=False)
        # if self.last_loss1 is None:
        #     self.last_loss1 = losses
        #     self.last_loss2 = losses
        #     return
        self.sync_level = 0
        # std1 = losses.std()
        # lastmean1 = self.last_loss1.mean()
        # lastmean2 = self.last_loss2.mean()
        # percdiff1 = torch.abs(losses.mean() - lastmean1) / lastmean1
        # percdiff2 = torch.abs(losses.mean() - lastmean2) / lastmean2
        # perc_spread = (losses.max() - losses.min()) / losses.mean()
        # # std = losses.std()
        # laststd1 = self.last_loss1.std()
        # laststd2 = self.last_loss2.std()
        # if self.rank == 0:
        #     print(perc_spread, percdiff1, percdiff2)
        # # if percdiff1 > 0.05 and std > 1.0:
        # if perc_spread > 0.02:
        #     self.sync_level = 1
        #     self.last_loss1 = losses
        # if perc_spread > 0.05:
        #     # self.tosync = True
        #     self.sync_level = 2
        #     self.last_loss1 = losses
        #     self.last_loss2 = losses
        # self.tosync = torch.tensor(0).cuda()
        # # c = 0
        # for n, p in self.network.named_parameters():
        #     if not p.requires_grad:
        #         continue
        #     norm = torch.linalg.norm(p.grad, )  # todo: order?
        #     # if self.rank == 0:
        #     if norm > 2:
        #         print(f"{n}: {norm.item():.5f}")
        #         self.tosync += 1
        #         break
        #     # if self.last_synch_benchmark[n] is None:
        #     #     self.last_synch_benchmark[n] = norm
        #     # elif norm >= self.last_synch_benchmark[n]:
        # self.to_sync_wait = dist.all_reduce(self.tosync, async_op=True)

    @torch.no_grad()
    def start_qsend(self):
        if self.to_sync_wait is not None:
            self.to_sync_wait.wait()
        if self.tosync > 0:
            if self.rank == 0:
                print("synching QRs")
        else:
            self.sending = False
            return
        # todo: linear, conv2d, batchnorm, more?
        # fwr = u @ s @ v.T  # full weight representation
        # conv weights -> out_ch, in_ch / groups, *kern size
        # bias - 1D (output shape)
        #   make this into a diagonal, then do the projection?
        #   can do 1/bias for the inverse
        # if weights.ndim == 1: average weights
        # linear - 2D weights
        self.sending = True
        self.sendqs = {}
        self.holdingrs = {}
        for n, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True)
                continue

            weights = p.data

            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights

            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            if lpweights.shape[0] >= lpweights.shape[1]:  # already TS of similar
                localq, localr = torch.linalg.qr(lpweights, mode='reduced')
                # trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode='reduced')
                # trans = True
            localq = localq.contiguous()
            # dist.all_reduce(localr, dist.ReduceOp.AVG, async_op=True)
            waitq = dist.all_reduce(localq, dist.ReduceOp.AVG, async_op=True)
            self.sendqs[n] = [localq, waitq]
            self.holdingrs[n] = localr

    @torch.no_grad()
    def update_after_send(self):
        for n, p in self.network.named_parameters():
            if not p.requires_grad or p.ndim == 1:
                continue

            weights = p.data
            shp = p.data.shape
            if weights.ndim > 2:
                lpweights = weights.view(weights.shape[0], -1)
            else:
                lpweights = weights
            self.sendqs[n][-1].wait()
            localq = self.sendqs[n][0]
            localr = self.holdingrs[n]
            # Q0 = P x Qlocal -> P = Q0 x Qlocal.T
            new = localq @ localr  # ) / dist.get_world_size()
            if shp[0] >= lpweights.shape[1]:  # already TS of similar
                p.data.set_(new.T.reshape(shp).contiguous())
            else:
                p.data.set_(new.reshape(shp).contiguous())

            # print(f"{compare.mean().item():.5f}, {compare.std().item():.5f}, {compare.min().item():.5f}, "
            #       f"{compare.max().item():.5f}")

            dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True)

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
                localq, localr = torch.linalg.qr(lpweights, mode='reduced')
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode='reduced')
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
    def project_weights(self, sync_level='all'):
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
                localq, localr = torch.linalg.qr(lpweights, mode='reduced')
                trans = False
            else:
                localq, localr = torch.linalg.qr(lpweights.T, mode='reduced')
                trans = True
            # if sync_level in ['r', 'all']:  # sync only R
            if sync_level == 'r':  # sync only R
                if self.param_buffers[n]['r'] is None:
                    self.param_buffers[n]['r'] = localr.contiguous()
                else:
                    self.param_buffers[n]['r'].zero_()
                    self.param_buffers[n]['r'].add_(localr)  # need contiguous tensor

                if self.param_buffers[n]['q'] is None:
                    self.param_buffers[n]['q'] = localq.contiguous()
                else:
                    self.param_buffers[n]['q'].zero_()
                    self.param_buffers[n]['q'].add_(localq)

                wait2[n] = dist.all_reduce(
                    self.param_buffers[n]['r'], dist.ReduceOp.AVG, async_op=True
                )
            elif sync_level in ['q', 'all']:
            # elif sync_level == 'q':
                if self.param_buffers[n]['r'] is None:
                    self.param_buffers[n]['r'] = localr.contiguous()
                else:
                    self.param_buffers[n]['r'].zero_()
                    self.param_buffers[n]['r'].add_(localr)  # need contiguous tensor

                if self.param_buffers[n]['q'] is None:
                    self.param_buffers[n]['q'] = localq.contiguous()
                else:
                    self.param_buffers[n]['q'].zero_()
                    self.param_buffers[n]['q'].add_(localq)

                wait2[n] = dist.all_reduce(
                    self.param_buffers[n]['q'], dist.ReduceOp.AVG, async_op=True
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
            new = self.param_buffers[n]['q'] @ self.param_buffers[n]['r']
            if trans:
                p.data.set_(new.T.view(shp))#.contiguous())
            else:
                p.data.set_(new.view(shp))#.contiguous())

            if sync_level == "all":
                # p.data.set_(p.data.contiguous())
                waits.append(dist.all_reduce(p.data, dist.ReduceOp.AVG, async_op=True))
        for w in waits:
            w.wait()


@torch.no_grad()
def project_weights(network: nn.Module):

    rank = dist.get_rank()
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
            localq, localr = torch.linalg.qr(lpweights, mode='reduced')
            trans = False
        else:
            localq, localr = torch.linalg.qr(lpweights.T, mode='reduced')
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

    rank = dist.get_rank()
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
        localq, localr = torch.linalg.qr(lpgrads.T, mode='complete')
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