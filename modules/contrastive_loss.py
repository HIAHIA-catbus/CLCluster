import torch
import torch.nn as nn
import math
import numpy as np


# BarlowTwins loss
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
class InstanceLoss(nn.Module):
    def __init__(self,args):
        super(InstanceLoss, self).__init__()

        self.args = args
    def __call__(self, z1, z2):
        # empirical cross-correlation matrix
        c = z1.T @ z2

        # # sum the cross-correlation matrix between all gpus
        # c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        # print("on_diag:",on_diag)
        # print("off_diag:",off_diag)
        loss = 0.0002*on_diag + self.args.lambd * off_diag
        return loss