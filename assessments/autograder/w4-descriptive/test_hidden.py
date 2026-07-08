import numpy as np
import pytest
from solution import describe


# ── original smoke test ───────────────────────────────────────────────────────

def test_describe():
    d = describe([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(d["mean"], 2.5)
    assert np.isclose(d["var"], 1.25)          # population variance
    assert np.isclose(d["std"], np.sqrt(1.25))
    assert np.isclose(d["median"], 2.5)


# ── oracle tests: numpy as ground truth, multiple sizes ──────────────────────

@pytest.mark.parametrize("seed,n", [(0, 20), (1, 50), (42, 7), (99, 100)])
def test_describe_oracle_random(seed, n):
    """All four statistics must match numpy on random normally-distributed data."""
    rng = np.random.default_rng(seed)
    x = rng.normal(10.0, 3.0, n)
    d = describe(x)
    assert np.isclose(d["mean"],   np.mean(x)),              f"mean mismatch seed={seed}"
    assert np.isclose(d["var"],    np.var(x, ddof=0)),       f"var (ddof=0) mismatch seed={seed}"
    assert np.isclose(d["std"],    np.std(x, ddof=0)),       f"std (ddof=0) mismatch seed={seed}"
    assert np.isclose(d["median"], np.median(x)),            f"median mismatch seed={seed}"


@pytest.mark.parametrize("seed,n", [(5, 30), (17, 80)])
def test_describe_oracle_uniform(seed, n):
    """Stats must be correct for uniform data (different distribution shape)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n)
    d = describe(x)
    assert np.isclose(d["mean"],   np.mean(x))
    assert np.isclose(d["var"],    np.var(x, ddof=0))
    assert np.isclose(d["std"],    np.std(x, ddof=0))
    assert np.isclose(d["median"], np.median(x))


# ── ddof trap: population vs sample variance ──────────────────────────────────

def test_describe_population_variance_not_sample():
    """var must use ddof=0 (population), NOT ddof=1 (sample)."""
    x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    d = describe(x)
    pop_var = np.var(x, ddof=0)    # = 4.0
    sample_var = np.var(x, ddof=1) # = 4.571...
    assert np.isclose(d["var"], pop_var), \
        f"expected population var {pop_var}, got {d['var']} (sample var is {sample_var})"
    # std must be consistent with the var
    assert np.isclose(d["std"], np.sqrt(pop_var))


# ── edge cases ────────────────────────────────────────────────────────────────

def test_describe_all_same():
    """All identical values → var=0, std=0, mean=median=value."""
    d = describe([7.0, 7.0, 7.0, 7.0])
    assert np.isclose(d["mean"],   7.0)
    assert np.isclose(d["var"],    0.0)
    assert np.isclose(d["std"],    0.0)
    assert np.isclose(d["median"], 7.0)


def test_describe_odd_length_median():
    """Odd-length array: median is the middle element."""
    x = [10.0, 20.0, 30.0]
    d = describe(x)
    assert np.isclose(d["median"], 20.0)


def test_describe_even_length_median():
    """Even-length array: median is the average of the two middle elements."""
    x = [1.0, 3.0, 5.0, 7.0]
    d = describe(x)
    assert np.isclose(d["median"], 4.0)


# ── trap: constant / hardcoded return values ──────────────────────────────────

def test_describe_mean_is_data_dependent():
    """Two different inputs must produce different means."""
    d1 = describe([1.0, 2.0, 3.0])
    d2 = describe([10.0, 20.0, 30.0])
    assert not np.isclose(d1["mean"], d2["mean"]), \
        "mean must reflect the actual data, not a constant"


def test_describe_std_is_data_dependent():
    """Two distributions with different spreads must give different std."""
    rng = np.random.default_rng(33)
    x_narrow = rng.normal(0, 0.1, 40)
    x_wide   = rng.normal(0, 10.0, 40)
    d_narrow = describe(x_narrow)
    d_wide   = describe(x_wide)
    assert d_narrow["std"] < d_wide["std"], \
        "std must be smaller for narrower data"


def test_describe_returns_required_keys():
    """Result dict must contain exactly mean, var, std, median."""
    d = describe([1.0, 2.0, 3.0])
    for key in ("mean", "var", "std", "median"):
        assert key in d, f"missing key '{key}'"
