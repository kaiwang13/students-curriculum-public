import numpy as np


def kfold_indices(n: int, k: int, seed: int = 0):
    """返回 k 折的 (train_idx, test_idx) 列表，测试折两两不相交且并集为全部样本。"""
    raise NotImplementedError
