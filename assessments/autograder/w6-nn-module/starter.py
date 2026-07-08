import torch
import torch.nn as nn


class MLP(nn.Module):
    """两层 MLP：Linear(in,hidden)->ReLU->Linear(hidden,out)。"""
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
