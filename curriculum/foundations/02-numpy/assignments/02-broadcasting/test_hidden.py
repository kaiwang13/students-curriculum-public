import numpy as np
from scipy.stats import zscore as scipy_zscore
from solution import standardize_columns


def _oracle(X):
    """Independent oracle: scipy zscore (ddof=0) per column."""
    return scipy_zscore(X, axis=0)


def test_columns_standardized():
    """Original test: keep for backward compat."""
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    Z = standardize_columns(X)
    assert Z.shape == X.shape
    assert np.allclose(Z.mean(axis=0), 0)
    assert np.allclose(Z.std(axis=0), 1)


def test_random_shapes_vs_oracle():
    """Multiple shapes/seeds: exact match against scipy zscore."""
    for seed, shape in [(0, (10, 3)), (1, (50, 5)), (5, (8, 7))]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=shape)
        Z = standardize_columns(X)
        expected = _oracle(X)
        assert Z.shape == X.shape, f"Shape mismatch seed={seed}"
        assert np.allclose(Z, expected, atol=1e-10), (
            f"Value mismatch seed={seed} shape={shape}"
        )


def test_column_means_zero():
    """Each column mean is 0 for random input."""
    rng = np.random.default_rng(42)
    X = rng.normal(scale=10.0, size=(100, 6))
    Z = standardize_columns(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-10)


def test_column_stds_one():
    """Each column std is 1 for random input."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-5, 5, size=(50, 4))
    Z = standardize_columns(X)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-10)


def test_large_scale_columns():
    """Columns with very different scales: both become std=1."""
    rng = np.random.default_rng(21)
    X = np.column_stack([rng.normal(0, 1, 40), rng.normal(0, 1000, 40)])
    Z = standardize_columns(X)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-10)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-10)


def test_trap_not_zeros():
    """Trap: returning all-zeros must fail when input has variance."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(20, 3))
    Z = standardize_columns(X)
    assert not np.allclose(Z, 0.0), "Non-trivial input must not produce all-zero output"


def test_exact_values_random():
    """Exact value check vs scipy oracle for a fresh random matrix."""
    rng = np.random.default_rng(55)
    X = rng.normal(size=(15, 4))
    assert np.allclose(standardize_columns(X), _oracle(X), atol=1e-10)
