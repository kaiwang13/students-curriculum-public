import numpy as np
from solution import genomic_lambda, qq_points


def test_lambda_null_is_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)             # null: uniform p
    assert abs(genomic_lambda(p) - 1.0) < 0.1


def test_lambda_inflated():
    rng = np.random.default_rng(1)
    chi2 = rng.chisquare(df=1, size=5000) * 1.5   # inflated
    from scipy import stats
    p = stats.chi2.sf(chi2, df=1)
    assert genomic_lambda(p) > 1.2


def test_qq_shapes():
    p = np.linspace(1e-6, 1, 100)
    e, o = qq_points(p)
    assert e.shape == (100,) and o.shape == (100,)
    assert np.all(np.diff(e) >= 0)          # expected sorted ascending
