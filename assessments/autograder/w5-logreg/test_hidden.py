import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from solution import fit_predict_proba


def test_separable_auc():
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5,
                               random_state=0)
    Xtr, Xte, ytr, yte = X[:300], X[300:], y[:300], y[300:]
    proba = fit_predict_proba(Xtr, ytr, Xte)
    assert proba.shape == (100,)
    assert roc_auc_score(yte, proba) > 0.80
