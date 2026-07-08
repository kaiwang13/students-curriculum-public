import numpy as np
from scipy import stats
from solution import rank_int


def test_int_is_normalish_and_monotone():
    rng = np.random.default_rng(0)
    x = rng.exponential(size=500)           # skewed input
    z = rank_int(x)
    assert abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.1
    order_x = np.argsort(x); order_z = np.argsort(z)
    assert np.array_equal(order_x, order_z)  # rank-preserving
    # Shapiro-ish: transformed data far more normal than the raw exponential
    assert stats.kurtosis(z) < stats.kurtosis(x)
