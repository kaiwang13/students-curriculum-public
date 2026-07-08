import numpy as np
from sklearn.linear_model import Ridge
from solution import ridge_predict


def test_matches_sklearn_ridge():
    rng = np.random.default_rng(0)
    G = rng.integers(0, 3, size=(80, 20)).astype(float)
    y = rng.normal(size=80)
    Gs = (G - G.mean(0)) / (G.std(0) + 1e-8)
    ref = Ridge(alpha=1.0, fit_intercept=True).fit(Gs, y).predict(Gs)
    assert np.allclose(ridge_predict(G, y, alpha=1.0), ref, atol=1e-6)


def test_reduces_residual():
    rng = np.random.default_rng(1)
    G = rng.integers(0, 3, size=(100, 10)).astype(float)
    y = G[:, 0] * 0.5 + rng.normal(scale=0.5, size=100)
    pred = ridge_predict(G, y, alpha=0.1)
    assert ((y - pred) ** 2).mean() < ((y - y.mean()) ** 2).mean()
