from __future__ import annotations

import torch.nn as nn


class DLRTModule(nn.Module):
    # parent class to abstract some methods
    def __init__(self):
        super().__init__()
        self.train_case = None
        self.dlrt = True
        self.prev_case = "s"

    def kl_postpro_s_prepro(self):
        # to be overwritten
        pass

    def kl_prepro(self):
        # to be overwritten
        pass

    def change_training_case(self, case):
        # switch -> if current train case is k/l, do post for
        if case not in ["k", "l", "s"]:
            raise ValueError(f"case must be one of k, l, or s, not: {case}")
        self.train_case = case

    def cycle_training_case(self):
        # make sure that everything runs from k -> l -> s
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
