import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """返回 auc、accuracy 及混淆矩阵四格 (tp,fp,tn,fn)。"""
    raise NotImplementedError
