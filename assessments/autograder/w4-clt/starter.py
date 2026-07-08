import numpy as np


def standard_error(x) -> float:
    """样本均值的标准误：s/√n（s 用 ddof=1）。"""
    raise NotImplementedError


def sample_means(data, n: int, reps: int, seed: int = 0):
    """有放回抽样 reps 次，每次 n 个，返回每次的样本均值（长度 reps 的数组）。"""
    raise NotImplementedError
