from __future__ import annotations

import torch
import torch.nn as nn


class DLRTModule(nn.Module):
    # parent class to abstract some methods
    def __init__(self, fixed=False):
        super().__init__()
        self.train_case = None
        self.dlrt = True
        self.prev_case = "s"
        self.fixed = fixed

    def _k_preprocess(self):
        ...

    def _k_postprocess(self):
        ...

    def _l_preprocess(self):
        ...

    def _l_postprocess(self):
        ...

    def _s_preprocess(self):
        ...

    @torch.no_grad()
    def kl_postpro_s_prepro(self):
        # ------- k postpro ----------------------------------
        self._k_postprocess()
        # ------- l postpro ----------------------------------
        self._l_postprocess()
        # ------- s prepro ------------------------------------
        # bias is trainable for the s step
        self._s_preprocess()

    @torch.no_grad()
    def kl_prepro(self):
        # disable bias training
        self._k_preprocess()
        self._l_preprocess()

    def rank_adaption(self):
        # to be overwritten (if needed in the not-fixed case)
        ...

    def change_training_case(self, case):
        # switch -> if current train case is k/l, do post for
        if case not in ["k", "l", "s"]:
            raise ValueError(f"case must be one of k, l, or s, not: {case}")
        self.train_case = case

    def cycle_training_case(self):
        # make sure that everything runs from k -> l -> s
        self.train()
        if self.train_case == "k" and self.prev_case == "s":
            self.train_case = "l"
        elif self.train_case == "l" and self.prev_case == "k":
            self.kl_postpro_s_prepro()
            self.train_case = "s"
        elif self.train_case == "s" and self.prev_case == "l":
            self.kl_prepro()
            self.train_case = "k"
        else:
            raise RuntimeError(
                f"Training case is not correct! last train case: {self.train_case}"
                f" 2x previous: {self.prev_case}",
            )
