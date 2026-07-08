import numpy as np
from scipy import stats


def assoc(G, y, covariates=None):
    """对每个 SNP 做单变量线性回归；协变量用 Frisch–Waugh–Lovell 残差化后再回归。
    G:(n,m) 基因型; y:(n,); covariates:(n,k) 或 None（截距总是包含）。
    返回 {'beta','se','t','p'}，每个长度 m。"""
    raise NotImplementedError
