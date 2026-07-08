import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from solution import binary_metrics


# ── original smoke test ───────────────────────────────────────────────────────

def test_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    m = binary_metrics(y_true, y_score)
    assert np.isclose(m["auc"], 1.0)
    assert np.isclose(m["accuracy"], 1.0)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 0, 0)


# ── oracle tests: sklearn as ground truth on randomised inputs ────────────────

@pytest.mark.parametrize("seed,n", [(17, 200), (42, 500), (7, 100), (99, 300)])
def test_metrics_oracle_random(seed, n):
    """AUC, accuracy, and confusion-matrix cells must match sklearn on random data."""
    rng = np.random.default_rng(seed)
    y_true  = rng.integers(0, 2, n)
    y_score = rng.uniform(0, 1, n)
    m = binary_metrics(y_true, y_score, threshold=0.5)

    ref_auc = roc_auc_score(y_true, y_score)
    y_pred  = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ref_acc = (tp + tn) / n

    assert np.isclose(m["auc"],      ref_auc), f"AUC mismatch seed={seed}"
    assert np.isclose(m["accuracy"], ref_acc), f"accuracy mismatch seed={seed}"
    assert m["tp"] == int(tp),  f"tp mismatch seed={seed}"
    assert m["fp"] == int(fp),  f"fp mismatch seed={seed}"
    assert m["tn"] == int(tn),  f"tn mismatch seed={seed}"
    assert m["fn"] == int(fn),  f"fn mismatch seed={seed}"


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_metrics_oracle_multiple_thresholds(threshold):
    """Confusion-matrix and accuracy must reflect the actual threshold used."""
    rng = np.random.default_rng(55)
    n = 200
    y_true  = rng.integers(0, 2, n)
    y_score = rng.uniform(0, 1, n)
    m = binary_metrics(y_true, y_score, threshold=threshold)

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ref_acc = (tp + tn) / n

    assert np.isclose(m["accuracy"], ref_acc), f"accuracy wrong at threshold={threshold}"
    assert m["tp"] == int(tp)
    assert m["fp"] == int(fp)
    assert m["tn"] == int(tn)
    assert m["fn"] == int(fn)


def test_metrics_class_imbalance():
    """Heavily imbalanced labels: AUC must still be computed correctly."""
    rng = np.random.default_rng(13)
    y_true  = np.array([0] * 90 + [1] * 10)
    y_score = rng.uniform(0, 1, 100)
    m = binary_metrics(y_true, y_score)
    ref_auc = roc_auc_score(y_true, y_score)
    assert np.isclose(m["auc"], ref_auc)


# ── confusion-matrix consistency invariants ───────────────────────────────────

@pytest.mark.parametrize("seed", [21, 34])
def test_confusion_matrix_sums_to_n(seed):
    """tp+fp+tn+fn must equal n (no elements lost or double-counted)."""
    rng = np.random.default_rng(seed)
    n = 150
    y_true  = rng.integers(0, 2, n)
    y_score = rng.uniform(0, 1, n)
    m = binary_metrics(y_true, y_score)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == n


def test_accuracy_consistent_with_confusion_matrix():
    """accuracy must equal (tp+tn)/n, not some independently computed value."""
    rng = np.random.default_rng(88)
    n = 120
    y_true  = rng.integers(0, 2, n)
    y_score = rng.uniform(0, 1, n)
    m = binary_metrics(y_true, y_score, threshold=0.5)
    computed_acc = (m["tp"] + m["tn"]) / n
    assert np.isclose(m["accuracy"], computed_acc), \
        "accuracy must equal (tp+tn)/n from the same confusion matrix"


# ── trap: constant score ──────────────────────────────────────────────────────

def test_trap_constant_score_auc():
    """Constant score → AUC ≈ 0.5, NOT 1.0 or 0.0."""
    rng = np.random.default_rng(77)
    y_true = rng.integers(0, 2, 100)
    y_const = np.full(100, 0.5)
    m = binary_metrics(y_true, y_const)
    assert np.isclose(m["auc"], 0.5, atol=0.01), \
        f"constant score should give AUC≈0.5, got {m['auc']}"


def test_trap_auc_is_not_accuracy():
    """AUC must differ from accuracy for non-perfect classifiers."""
    rng = np.random.default_rng(66)
    n = 200
    y_true  = rng.integers(0, 2, n)
    y_score = rng.uniform(0, 1, n)   # random, not calibrated
    m = binary_metrics(y_true, y_score)
    # They are rarely equal on random data; checking they are computed independently
    ref_auc = roc_auc_score(y_true, y_score)
    assert np.isclose(m["auc"], ref_auc), "AUC is wrong"


def test_trap_threshold_affects_confusion_not_auc():
    """Changing threshold must change tp/fp/tn/fn but NOT auc."""
    rng = np.random.default_rng(44)
    y_true  = rng.integers(0, 2, 200)
    y_score = rng.uniform(0, 1, 200)
    m_low  = binary_metrics(y_true, y_score, threshold=0.3)
    m_high = binary_metrics(y_true, y_score, threshold=0.7)
    assert np.isclose(m_low["auc"], m_high["auc"]), "AUC must not depend on threshold"
    assert m_low["tp"] != m_high["tp"], "tp must change with threshold"
