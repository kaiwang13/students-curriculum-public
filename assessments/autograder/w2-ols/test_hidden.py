import numpy as np
from solution import ols_beta


def _oracle(X, y):
    """Independent oracle: lstsq (minimum-norm least-squares solution)."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def test_recovers_known_beta():
    """Original test: keep for backward compat."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta
    assert np.allclose(ols_beta(X, y), beta, atol=1e-8)


def test_random_beta_multiple_seeds():
    """Random beta and X for several seeds; compare against lstsq oracle."""
    for seed in [1, 2, 5, 9]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(150, 4))
        beta_true = rng.normal(size=(4,))
        # tiny noise so lstsq and solve agree closely
        y = X @ beta_true + rng.normal(scale=1e-3, size=(150,))
        expected = _oracle(X, y)
        got = ols_beta(X, y)
        assert np.allclose(got, expected, atol=1e-5), (
            f"seed={seed}: got {got}, expected {expected}"
        )


def test_different_shapes():
    """(n=50, p=2) and (n=500, p=10): shape and value checks."""
    for seed, n, p in [(3, 50, 2), (4, 500, 10)]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        y = rng.normal(size=(n,))
        expected = _oracle(X, y)
        got = ols_beta(X, y)
        assert got.shape == (p,), f"Shape mismatch: {got.shape} vs ({p},)"
        assert np.allclose(got, expected, atol=1e-6), (
            f"seed={seed} n={n} p={p}"
        )


def test_noiseless_recovery():
    """Exact noiseless recovery: beta reconstructed to near machine precision."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 5))
    beta = np.array([3.0, -1.5, 0.0, 2.0, -0.5])
    y = X @ beta
    assert np.allclose(ols_beta(X, y), beta, atol=1e-8)


def test_output_shape():
    """Output must be 1-D with length p."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(80, 6))
    y = rng.normal(size=(80,))
    beta = ols_beta(X, y)
    assert beta.shape == (6,), f"Expected shape (6,), got {beta.shape}"


def test_trap_not_zeros():
    """Trap: zero vector must fail when true beta is non-trivial."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(100, 3))
    beta_true = np.array([5.0, -3.0, 1.0])
    y = X @ beta_true
    got = ols_beta(X, y)
    assert not np.allclose(got, 0.0), "Beta should not be all-zeros"
    assert np.allclose(got, beta_true, atol=1e-8)


def test_minimises_residuals():
    """OLS beta minimises ||y - X beta||: perturbation increases loss."""
    rng = np.random.default_rng(23)
    X = rng.normal(size=(120, 4))
    y = rng.normal(size=(120,))
    beta = ols_beta(X, y)
    loss = np.sum((y - X @ beta) ** 2)
    perturb = beta + rng.normal(scale=0.1, size=beta.shape)
    loss_perturbed = np.sum((y - X @ perturb) ** 2)
    assert loss < loss_perturbed, "OLS beta must minimise squared residuals"
