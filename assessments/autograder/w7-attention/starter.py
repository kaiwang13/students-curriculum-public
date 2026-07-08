import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """缩放点积注意力。q,k,v: (..., T, d)。返回 (输出, 注意力权重)。
    mask: 可选 (T, T) 布尔/0-1，0 处不允许注意（置 -inf）。"""
    raise NotImplementedError
