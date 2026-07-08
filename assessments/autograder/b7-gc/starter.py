import numpy as np
from scipy import stats


def genomic_lambda(pvals):
    """基因组膨胀因子 λ_GC = median(观测 χ²) / median(χ²_1df 理论)。"""
    raise NotImplementedError


def qq_points(pvals):
    """QQ 图数据：返回 (期望 -log10p, 观测 -log10p)，均升序。"""
    raise NotImplementedError
