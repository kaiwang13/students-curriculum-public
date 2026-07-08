import numpy as np
from solution import pca_transform


def test_pca_shapes_and_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    scores, evr = pca_transform(X, 3)
    assert scores.shape == (200, 3)
    assert evr.shape == (3,)
    assert np.all(np.diff(evr) <= 1e-9)      # ratios are non-increasing
    assert 0 < evr.sum() <= 1 + 1e-9
