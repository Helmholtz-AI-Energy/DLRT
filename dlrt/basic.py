from __future__ import annotations

import torch.nn as nn

__all__ = ["DLRTModule"]


class DLRTModule(nn.Module):
    # parent class to abstract some methods
    def __init__(self, fixed=False):
        super().__init__()
        self.train_case = None
        self.dlrt = True
        self.prev_case = "s"
        self.fixed = fixed
        self.basic_number_weights = None

    def k_preprocess(self):
        ...

    def k_postprocess(self):
        ...

    def l_preprocess(self):
        ...

    def l_postprocess(self):
        ...

    def s_preprocess(self):
        ...

    def rank_adaption(self):
        # to be overwritten (if needed in the not-fixed case)
        ...

    def get_rank_percentage(self):
        """
        Get the percentage of ranks being used compared to the number of weights in a dense layer
        """
        return f"{self.low_rank / self.basic_number_weights:.5f}, {self.low_rank}"

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
