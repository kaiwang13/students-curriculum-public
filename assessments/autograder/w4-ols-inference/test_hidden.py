import numpy as np
from scipy import stats
from solution import ols_inference


def test_matches_scipy_linregress():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=60)
    r = ols_inference(x, y)
    lr = stats.linregress(x, y)
    assert np.isclose(r["beta1"], lr.slope)
    assert np.isclose(r["se1"], lr.stderr)
    assert np.isclose(r["p1"], lr.pvalue)
