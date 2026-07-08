import numpy as np
from solution import gram


def _oracle(X):
    """Independent oracle: Gram matrix via matmul (no einsum)."""
    return np.matmul(X, X.T)


def test_gram():
    """Original test: keep for backward compat."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    G = gram(X)
    assert G.shape == (3, 3)
    assert np.allclose(G, X @ X.T)


def test_gram_output_shape():
    """Output shape is (n, n) for (n, d) input."""
    for seed, n, d in [(0, 5, 3), (1, 10, 7), (2, 3, 100)]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, d))
        G = gram(X)
        assert G.shape == (n, n), f"Expected ({n},{n}), got {G.shape}"


def test_gram_values_random_seeds():
    """Values match matmul oracle on multiple random inputs."""
    for seed in [3, 7, 42]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(8, 5))
        assert np.allclose(gram(X), _oracle(X), atol=1e-10), (
            f"Value mismatch seed={seed}"
        )


def test_gram_symmetric():
    """Gram matrix must be symmetric: G == G.T."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(6, 4))
    G = gram(X)
    assert np.allclose(G, G.T, atol=1e-10)


def test_gram_diagonal_is_squared_norm():
    """Diagonal G[i,i] must equal ||X[i]||^2."""
    rng = np.random.default_rng(22)
    X = rng.normal(size=(7, 5))
    G = gram(X)
    expected_diag = np.sum(X ** 2, axis=1)
    assert np.allclose(np.diag(G), expected_diag, atol=1e-10)


def test_gram_psd():
    """Gram matrix must be positive semi-definite (all eigenvalues >= 0)."""
    rng = np.random.default_rng(33)
    X = rng.normal(size=(5, 3))
    G = gram(X)
    eigvals = np.linalg.eigvalsh(G)
    assert np.all(eigvals >= -1e-9), f"PSD violated: min eigenvalue {eigvals.min()}"


def test_gram_trap_not_identity():
    """Trap: returning identity matrix must fail for non-trivial input."""
    rng = np.random.default_rng(44)
    X = rng.normal(size=(4, 4))
    G = gram(X)
    expected = _oracle(X)
    assert not np.allclose(G, np.eye(4)), "Result should not be identity matrix"
    assert np.allclose(G, expected, atol=1e-10)


def test_gram_wide_matrix():
    """n < d (wide matrix): Gram still has shape (n, n)."""
    rng = np.random.default_rng(55)
    X = rng.normal(size=(3, 20))
    G = gram(X)
    assert G.shape == (3, 3)
    assert np.allclose(G, _oracle(X), atol=1e-10)
