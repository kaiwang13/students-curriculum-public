import numpy as np


def ridge_predict(G, y, alpha=1.0):
    """全基因组岭回归（LMM 的多基因随机效应近似）：用所有 SNP 预测 y，返回预测值 (n,)。
    G 先标准化；y 去均值后闭式解 β=(GᵀG+αI)⁻¹Gᵀ(y-ȳ)，预测 = ȳ + Gs·β（含截距）。"""
    raise NotImplementedError
