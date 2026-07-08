import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from solution import fit_xgb_auc


# ── original smoke test ───────────────────────────────────────────────────────

def test_xgb_learns():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                               random_state=1)
    Xtr, Xte, ytr, yte = X[:350], X[350:], y[:350], y[350:]
    auc = fit_xgb_auc(Xtr, ytr, Xte, yte)
    assert auc > 0.85


# ── larger / harder dataset ───────────────────────────────────────────────────

def test_xgb_larger_dataset():
    """AUC must exceed 0.88 on a larger, more informative dataset."""
    X, y = make_classification(n_samples=1000, n_features=15, n_informative=8,
                               random_state=42)
    Xtr, Xte = X[:700], X[700:]
    ytr, yte = y[:700], y[700:]
    auc = fit_xgb_auc(Xtr, ytr, Xte, yte)
    assert auc > 0.88, f"expected AUC > 0.88, got {auc:.4f}"


def test_xgb_returns_float_in_valid_range():
    """Return value must be a float in [0, 1]."""
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5,
                               random_state=7)
    Xtr, Xte = X[:280], X[280:]
    ytr, yte = y[:280], y[280:]
    auc = fit_xgb_auc(Xtr, ytr, Xte, yte)
    assert isinstance(auc, float), f"expected float, got {type(auc)}"
    assert 0.0 <= auc <= 1.0, f"AUC {auc} out of [0,1]"


# ── determinism with fixed seed ───────────────────────────────────────────────

def test_xgb_deterministic():
    """Calling fit_xgb_auc twice with the same data must return the same AUC."""
    X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                               random_state=1)
    Xtr, Xte = X[:350], X[350:]
    ytr, yte = y[:350], y[350:]
    auc1 = fit_xgb_auc(Xtr, ytr, Xte, yte)
    auc2 = fit_xgb_auc(Xtr, ytr, Xte, yte)
    assert auc1 == auc2, f"non-deterministic: got {auc1} then {auc2}"


# ── trap: naive / constant-return implementations ─────────────────────────────

def test_xgb_trap_not_constant_auc():
    """AUC must vary with the quality of the classifier (not hardcoded 1.0 or 0.5)."""
    # Easy dataset → high AUC
    X_easy, y_easy = make_classification(n_samples=600, n_features=10,
                                         n_informative=9, n_redundant=1,
                                         random_state=20)
    auc_easy = fit_xgb_auc(X_easy[:400], y_easy[:400],
                            X_easy[400:], y_easy[400:])

    # Hard dataset (few informative features) → lower AUC
    X_hard, y_hard = make_classification(n_samples=600, n_features=10,
                                         n_informative=1, n_redundant=0,
                                         n_clusters_per_class=1,
                                         random_state=21)
    auc_hard = fit_xgb_auc(X_hard[:400], y_hard[:400],
                            X_hard[400:], y_hard[400:])

    assert auc_easy > auc_hard, \
        f"easier dataset should yield higher AUC ({auc_easy:.3f} vs {auc_hard:.3f})"


def test_xgb_better_than_random():
    """AUC must be meaningfully above 0.5 on a learnable problem."""
    X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                               random_state=1)
    Xtr, Xte = X[:350], X[350:]
    ytr, yte = y[:350], y[350:]
    auc = fit_xgb_auc(Xtr, ytr, Xte, yte)
    assert auc > 0.7, f"AUC {auc:.3f} is suspiciously low — random guessing is 0.5"
