import numpy as np
from sklearn.decomposition import PCA


def genotype_pca(G, k):
    """标准化基因型后做 PCA，返回前 k 个主成分 (n, k)，用于群体结构校正。"""
    raise NotImplementedError
