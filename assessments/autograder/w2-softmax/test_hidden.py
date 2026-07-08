import numpy as np
from scipy.special import softmax as scipy_softmax
from solution import softmax


def _oracle(x, axis):
    return scipy_softmax(x, axis=axis)


def test_softmax_rows_sum_to_one():
    """Original test: keep for backward compat."""
    X = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    S = softmax(X, axis=1)
    assert np.allclose(S.sum(axis=1), 1)
    assert np.all(np.isfinite(S))
    assert np.allclose(S[1], [1 / 3, 1 / 3, 1 / 3])


def test_softmax_nonuniform_values():
    """Non-uniform rows: a constant-uniform implementation must fail."""
    rng = np.random.default_rng(42)
    X = rng.normal(scale=5.0, size=(10, 8))
    for ax in (0, 1, -1):
        expected = _oracle(X, ax)
        got = softmax(X, axis=ax)
        assert np.allclose(got, expected, atol=1e-10), f"axis={ax}: values mismatch"
        # rows (or cols) must NOT all be equal — non-uniform
        assert not np.allclose(got, got.mean(axis=ax, keepdims=True)), (
            f"axis={ax}: output must be non-uniform for non-uniform input"
        )


def test_softmax_random_seeds():
    """Several shapes/seeds checked against scipy oracle."""
    for seed, shape in [(1, (5, 4)), (2, (3, 10)), (7, (20, 6))]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=shape)
        for ax in (0, 1):
            assert np.allclose(softmax(X, axis=ax), _oracle(X, ax), atol=1e-10), (
                f"seed={seed} shape={shape} axis={ax}"
            )


def test_softmax_large_values_stability():
    """Values +1000: must be finite and match scipy."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4, 5)) + 1000.0
    expected = _oracle(X, -1)
    got = softmax(X, axis=-1)
    assert np.all(np.isfinite(got)), "Output must be finite for large inputs"
    assert np.allclose(got, expected, atol=1e-10)


def test_softmax_axis0():
    """Columns sum to 1 when axis=0; values match scipy."""
    rng = np.random.default_rng(99)
    X = rng.normal(size=(6, 4))
    got = softmax(X, axis=0)
    assert np.allclose(got.sum(axis=0), 1.0, atol=1e-10)
    assert np.allclose(got, _oracle(X, 0), atol=1e-10)


def test_softmax_1d_default_axis():
    """1-D array with default axis=-1 sums to 1 and matches scipy."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(8,))
    got = softmax(x)
    expected = _oracle(x, -1)
    assert np.allclose(got, expected, atol=1e-10)
    assert np.isclose(got.sum(), 1.0)


def test_softmax_trap_constant_vs_distinct_row():
    """Trap: a zero row gives uniform; a non-zero row must be non-uniform."""
    X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    S = softmax(X, axis=1)
    assert np.allclose(S[0], 1 / 3, atol=1e-10), "Uniform row must give uniform softmax"
    assert not np.allclose(S[1], 1 / 3), "Non-uniform row must give non-uniform softmax"
    assert np.allclose(S, _oracle(X, 1), atol=1e-10)
