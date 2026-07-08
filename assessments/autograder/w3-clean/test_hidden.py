"""Hidden tests for w3-clean.

Hardened: 100-row seeded frame with 3 numeric columns, ~10% NaNs each, 10
duplicate rows; independent median verification; median-vs-mean trap; index-
reset check; group uppercase trap; dtype preservation.
"""
import numpy as np
import pandas as pd
from solution import clean


def _make_dirty(seed=13, n=100, n_dupes=10):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18.0, 80.0, n).round(1)
    x1 = rng.uniform(15.0, 45.0, n).round(1)
    x2 = rng.uniform(90.0, 180.0, n).round(1)
    group = rng.choice(["a", "b", "A", "B"], n)
    # Inject NaNs (~10% each)
    for arr in (age, x1, x2):
        arr[rng.choice(n, n // 10, replace=False)] = np.nan
    df = pd.DataFrame({"age": age, "x1": x1, "x2": x2, "group": group})
    # Append duplicate rows
    dup_idx = rng.choice(n, n_dupes, replace=False)
    return pd.concat([df, df.iloc[dup_idx]], ignore_index=True)


def _deduped_median(df, col):
    """Median of *col* in the deduped frame, matching solution's order of ops."""
    return float(df.drop_duplicates()[col].median())


def test_clean_basic():
    """Original small regression test."""
    df = pd.DataFrame({
        "group": ["a", "B", "a"],
        "x1": [20.0, np.nan, 20.0],
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    out = clean(df)
    assert out.duplicated().sum() == 0
    assert out["x1"].isna().sum() == 0
    assert set(out["group"]) <= {"A", "B"}


def test_no_duplicates():
    df = _make_dirty()
    out = clean(df)
    assert out.duplicated().sum() == 0, "Duplicates remain after clean()"


def test_no_nans_in_numeric():
    df = _make_dirty()
    out = clean(df)
    numeric_cols = out.select_dtypes(include="number").columns
    assert out[numeric_cols].isna().sum().sum() == 0


def test_group_uppercase():
    df = _make_dirty()
    out = clean(df)
    assert set(out["group"]) <= {"A", "B"}, f"Unexpected group values: {set(out['group'])}"


def test_index_reset():
    df = _make_dirty()
    out = clean(df)
    assert list(out.index) == list(range(len(out))), "Index must be 0-based after reset"


def test_median_fill_value_age():
    """NaN in age must be filled with the post-dedup median, not mean or zero."""
    df = _make_dirty()
    expected_median = _deduped_median(df, "age")
    out = clean(df)
    assert expected_median in out["age"].values or np.isnan(expected_median), (
        "Expected median value not found in filled age column"
    )
    deduped = df.drop_duplicates()
    expected_mean = float(deduped["age"].mean())
    if not np.isclose(expected_median, expected_mean, rtol=1e-3):
        assert expected_mean not in out["age"].values or expected_median in out["age"].values


def test_median_not_mean_trap():
    """Explicit trap: [1,1,1,1,100] — median=1, mean=20.8; fill must be 1."""
    df = pd.DataFrame({
        "x": [1.0, 1.0, 1.0, 1.0, 100.0, np.nan],
        "group": ["a", "b", "a", "b", "a", "b"],
    })
    out = clean(df)
    assert out["x"].isna().sum() == 0
    filled_val = out.loc[out["x"].round(1) != 100.0, "x"]
    assert (filled_val == 1.0).all(), (
        f"Expected all non-100 values to be 1.0 (median), got {filled_val.tolist()}"
    )


def test_dtype_numeric_preserved():
    """Numeric column dtypes must remain float64 after filling."""
    df = _make_dirty()
    out = clean(df)
    for col in ["age", "x1", "x2"]:
        assert out[col].dtype == np.float64, (
            f"Column {col} dtype changed: expected float64, got {out[col].dtype}"
        )


def test_multiple_numeric_columns_all_filled():
    """All three numeric columns must be NaN-free."""
    df = _make_dirty(seed=99)
    out = clean(df)
    for col in ["age", "x1", "x2"]:
        assert out[col].isna().sum() == 0, f"NaN remains in {col}"


def test_output_row_count():
    """Row count after clean must equal unique rows in input."""
    df = _make_dirty()
    expected_rows = len(df.drop_duplicates())
    out = clean(df)
    assert len(out) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(out)}"
    )
