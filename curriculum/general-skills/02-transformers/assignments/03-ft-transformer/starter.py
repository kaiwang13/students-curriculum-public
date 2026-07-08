import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """FT-Transformer 特征分词器：每个数值特征映射到一个 d_token 维 token，并前置一个 [CLS] token。"""
    def __init__(self, n_features, d_token):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
