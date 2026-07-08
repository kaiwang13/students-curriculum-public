import numpy as np
from scipy.spatial.distance import cdist
from solution import standardize, pairwise_sq_dists


# ── oracles ──────────────────────────────────────────────────────────────────

def _std_oracle(x):
    """Global z-score computed independently."""
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / sigma


def _dist_oracle(a, b):
    """Pairwise squared distances via scipy cdist (independent library oracle)."""
    return cdist(np.asarray(a).reshape(-1, 1), np.asarray(b).reshape(-1, 1), "sqeuclidean")


# ── standardize ───────────────────────────────────────────────────────────────

def test_standardize():
    """Original test: keep for backward compat."""
    z = standardize(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(z.mean(), 0.0)
    assert np.isclose(z.std(), 1.0)
    assert np.allclose(z, [-1.22474487, 0.0, 1.22474487])


def test_standardize_random_inputs():
    """Random arrays: values match oracle, mean=0, std=1."""
    for seed in [0, 7, 42]:
        rng = np.random.default_rng(seed)
        x = rng.normal(scale=10.0, size=(50,))
        z = standardize(x)
        assert np.allclose(z, _std_oracle(x), atol=1e-10), f"seed={seed}"
        assert np.isclose(z.mean(), 0.0, atol=1e-10)
        assert np.isclose(z.std(), 1.0, atol=1e-10)


def test_standardize_larger_array():
    """200-element array: mean=0, std=1 via oracle."""
    rng = np.random.default_rng(99)
    x = rng.uniform(-100, 100, size=(200,))
    z = standardize(x)
    assert np.allclose(z, _std_oracle(x), atol=1e-10)
    assert np.isclose(z.mean(), 0.0, atol=1e-10)
    assert np.isclose(z.std(), 1.0, atol=1e-10)


def test_standardize_trap_not_constant():
    """Trap: returning a constant must fail when input has variance."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20,))
    z = standardize(x)
    assert not np.allclose(z, z[0]), "Output must not be a constant vector"


def test_standardize_trap_not_input():
    """Trap: returning input unchanged must fail (mean/std differ)."""
    rng = np.random.default_rng(31)
    x = rng.normal(loc=5.0, scale=3.0, size=(30,))
    z = standardize(x)
    assert not np.allclose(z, x), "Output must differ from input"
    assert np.isclose(z.mean(), 0.0, atol=1e-10)


# ── pairwise_sq_dists ─────────────────────────────────────────────────────────

def test_pairwise():
    """Original test: keep for backward compat."""
    d = pairwise_sq_dists(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert d.shape == (2, 2)
    assert np.allclose(d, [[0, 4], [1, 1]])


def test_pairwise_shape():
    """Output shape is (m, n) for inputs of length m and n."""
    for seed, m, n in [(1, 5, 8), (2, 10, 3), (3, 1, 7)]:
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(m,))
        b = rng.normal(size=(n,))
        d = pairwise_sq_dists(a, b)
        assert d.shape == (m, n), f"Expected ({m},{n}), got {d.shape}"


def test_pairwise_values_random():
    """Values match explicit outer-diff oracle on random inputs."""
    for seed in [4, 8, 15]:
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(6,))
        b = rng.normal(size=(7,))
        got = pairwise_sq_dists(a, b)
        expected = _dist_oracle(a, b)
        assert np.allclose(got, expected, atol=1e-10), f"seed={seed}"


def test_pairwise_nonnegative():
    """All squared distances must be >= 0."""
    rng = np.random.default_rng(99)
    a = rng.normal(size=(8,))
    b = rng.normal(size=(5,))
    assert np.all(pairwise_sq_dists(a, b) >= 0)


def test_pairwise_self_diagonal_zero():
    """When a == b, diagonal must be exactly 0."""
    rng = np.random.default_rng(66)
    a = rng.normal(size=(5,))
    d = pairwise_sq_dists(a, a)
    assert np.allclose(np.diag(d), 0.0, atol=1e-10)


def test_pairwise_trap_not_zeros():
    """Trap: all-zero output must fail for non-trivial a, b."""
    a = np.array([1.0, 3.0, 5.0])
    b = np.array([2.0, 4.0])
    d = pairwise_sq_dists(a, b)
    assert not np.allclose(d, 0.0), "Non-trivial inputs must not give all-zero output"
    assert np.allclose(d, _dist_oracle(a, b), atol=1e-10)
