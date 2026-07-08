import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def fit_xgb_auc(Xtr, ytr, Xte, yte) -> float:
    """训练 XGBoost 分类器（hist），返回测试集 ROC-AUC。"""
    raise NotImplementedError
