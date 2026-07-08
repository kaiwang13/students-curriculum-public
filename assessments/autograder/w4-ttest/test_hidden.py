import numpy as np
from scipy import stats
from solution import two_sample_t


def test_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 40)
    b = rng.normal(0.7, 1.0, 40)
    r = two_sample_t(a, b)
    t, p = stats.ttest_ind(a, b, equal_var=True)
    assert np.isclose(r["t"], t)
    assert np.isclose(r["p"], p)
