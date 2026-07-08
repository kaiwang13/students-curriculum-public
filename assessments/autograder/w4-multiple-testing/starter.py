import numpy as np


def bonferroni(pvals, alpha: float = 0.05):
    """Bonferroni 校正：返回布尔数组（是否拒绝原假设）。"""
    raise NotImplementedError


def bh_fdr(pvals, alpha: float = 0.05):
    """Benjamini–Hochberg FDR：返回布尔数组（保持原顺序）。"""
    raise NotImplementedError
