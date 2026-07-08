import numpy as np


def loco_residualize(G, y, chrom, target_chrom, alpha=1.0):
    """留一染色体 (LOCO)：用除 target_chrom 外所有染色体的 SNP 做岭回归预测 y，返回残差 (n,)。
    chrom: (n_snps,) 每个 SNP 的染色体标签。检验 target_chrom 上的变异时用此残差作表型。"""
    raise NotImplementedError
