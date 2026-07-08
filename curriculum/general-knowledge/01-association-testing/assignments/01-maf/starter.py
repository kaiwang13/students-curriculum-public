import numpy as np


def maf(G):
    """G: (n_samples, n_snps) 剂量 in {0,1,2}. 返回每个 SNP 的最小等位基因频率 (n_snps,)。"""
    raise NotImplementedError


def mac(G):
    """最小等位基因计数 minor allele count (n_snps,)。"""
    raise NotImplementedError
