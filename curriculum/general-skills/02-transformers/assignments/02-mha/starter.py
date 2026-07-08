import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头自注意力：投影 Q/K/V → 分头 → 缩放点积注意力 → 合并 → 输出投影。"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
