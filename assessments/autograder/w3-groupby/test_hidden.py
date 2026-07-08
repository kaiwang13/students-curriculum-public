"""Hidden tests for w3-groupby.

Hardened: 200 rows / 20 seeded groups, unsorted input, float means verified
independently via pure-Python dict accumulation (not pandas groupby), sort-order
trap, single-group edge, and mutation guard.
"""
import numpy as np
import pandas as pd
from solution import group_means


def _make_large(seed=42, n=200, n_groups=20):
    rng = np.random.default_rng(seed)
    labels = [chr(65 + i) for i in range(n_groups)]  # A..T
    g = rng.choice(labels, size=n)
    v = rng.uniform(-50.0, 150.0, size=n)
    # Shuffle to deliberately unsorted order
    idx = rng.permutation(n)
    return pd.DataFrame({"cat": g[idx], "val": v[idx]})


def _expected_means(df, key, value):
    """Compute expected group means via plain Python — independent of groupby."""
    acc: dict = {}
    cnt: dict = {}
    for g, v in zip(df[key], df[value]):
        acc[g] = acc.get(g, 0.0) + v
        cnt[g] = cnt.get(g, 0) + 1
    keys = sorted(acc.keys())
    return pd.DataFrame({key: keys, value: [acc[k] / cnt[k] for k in keys]})


def test_group_means_basic():
    """Original small regression test."""
    df = pd.DataFrame({"g": ["a", "b", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    out = group_means(df, "g", "v")
    assert list(out["g"]) == ["a", "b"]
    assert list(out["v"]) == [2.0, 3.0]


def test_group_means_shape():
    """Output must have exactly 20 rows (one per group) and 2 columns in order."""
    df = _make_large()
    out = group_means(df, "cat", "val")
    assert out.shape == (20, 2), f"Expected (20, 2), got {out.shape}"
    assert list(out.columns) == ["cat", "val"]


def test_group_means_sorted_ascending():
    """Output must be sorted ascending by key even when input is shuffled."""
    df = _make_large()
    out = group_means(df, "cat", "val")
    keys = out["cat"].tolist()
    assert keys == sorted(keys), f"Keys not sorted: {keys}"


def test_group_means_values_correct():
    """Mean values must match independent dict-accumulation computation (rtol 1e-10)."""
    df = _make_large()
    out = group_means(df, "cat", "val").reset_index(drop=True)
    exp = _expected_means(df, "cat", "val").reset_index(drop=True)
    pd.testing.assert_frame_equal(out, exp, check_exact=False, rtol=1e-10)


def test_group_means_unsorted_trap():
    """Reverse-sorted input — output must still be ascending with correct means."""
    df = pd.DataFrame({
        "g": ["z", "y", "x", "y", "x", "z"],
        "v": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    out = group_means(df, "g", "v")
    assert list(out["g"]) == ["x", "y", "z"], "Groups must be sorted ascending"
    # x: (30+50)/2=40, y: (20+40)/2=30, z: (10+60)/2=35
    np.testing.assert_allclose(out["v"].values, [40.0, 30.0, 35.0])


def test_group_means_different_seed():
    """Second seeded dataset — values must still match independent computation."""
    df = _make_large(seed=99, n=300, n_groups=15)
    out = group_means(df, "cat", "val").reset_index(drop=True)
    exp = _expected_means(df, "cat", "val").reset_index(drop=True)
    pd.testing.assert_frame_equal(out, exp, check_exact=False, rtol=1e-10)


def test_group_means_single_group():
    """Single-group edge case: result has one row with the overall mean."""
    df = pd.DataFrame({"g": ["only"] * 6, "v": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]})
    out = group_means(df, "g", "v")
    assert len(out) == 1
    np.testing.assert_allclose(out["v"].values, [6.0])


def test_group_means_no_mutation():
    """The function must not mutate its input DataFrame."""
    df = _make_large(seed=77)
    snapshot = df.copy(deep=True)
    group_means(df, "cat", "val")
    pd.testing.assert_frame_equal(df, snapshot)
