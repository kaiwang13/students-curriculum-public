"""Hidden tests for p-w3-data-pipeline.

Hardened: 200-row seeded synthetic-table frame + 20 duplicates, ~10% NaNs per numeric col;
row-count-after-dedup verified via independent drop_duplicates(); median fill
values verified independently via numpy nanmedian on pre-pipeline deduped data;
group normalization and index-reset checked; median-vs-mean trap included.

All expected values are computed WITHOUT calling build_pipeline, so the tests
do not self-verify.
"""
import numpy as np
import pandas as pd
from solution import build_pipeline

REQUIRED = ["age", "group", "x1", "x2", "x3", "label"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_messy() -> pd.DataFrame:
    """Original 10-row fixture — kept for regression."""
    df = pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age":       [45.0, 62.0, 33.0, 55.0, 48.0, 70.0, 28.0, 51.0, 39.0, 66.0],
        "group":     ["a", "B", "a", "b", "A", "b", "A", "b", "a", "B"],
        "x1":        [28.0, np.nan, 22.5, 30.1, np.nan, 25.0, 19.8, 33.0, 27.0, 24.5],
        "x2":        [130.0, 145.0, 118.0, np.nan, 135.0, 160.0, 112.0, 140.0, 125.0, 155.0],
        "x3":        [95.0, 110.0, np.nan, 88.0, 102.0, 120.0, 78.0, 98.0, 91.0, 115.0],
        "label":     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    dupes = df.iloc[:2].copy()
    return pd.concat([df, dupes], ignore_index=True)


def _make_large_messy(seed=17, n=200, n_dupes=20) -> pd.DataFrame:
    """200-row seeded synthetic-table frame with NaNs and duplicates."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(18.0, 85.0, n).round(1)
    x1 = rng.uniform(15.0, 45.0, n).round(1)
    x2 = rng.uniform(90.0, 200.0, n).round(1)
    x3 = rng.uniform(60.0, 200.0, n).round(1)
    group = rng.choice(["a", "b", "A", "B"], n)
    label = rng.integers(0, 2, n)
    # Inject ~10% NaNs into each numeric column
    for arr in (age, x1, x2, x3):
        arr[rng.choice(n, n // 10, replace=False)] = np.nan
    df = pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "age": age, "group": group, "x1": x1,
        "x2": x2, "x3": x3, "label": label,
    })
    dup_idx = rng.choice(n, n_dupes, replace=False)
    return pd.concat([df, df.iloc[dup_idx]], ignore_index=True)


# ---------------------------------------------------------------------------
# Original regression tests (kept intact)
# ---------------------------------------------------------------------------

def test_no_dupes():
    out = build_pipeline(_make_messy())
    assert out.duplicated().sum() == 0


def test_no_nans_in_numeric():
    out = build_pipeline(_make_messy())
    assert out.select_dtypes(include="number").isna().sum().sum() == 0


def test_group_normalized():
    out = build_pipeline(_make_messy())
    assert set(out["group"]) <= {"A", "B"}


def test_parquet_roundtrip(tmp_path):
    out = build_pipeline(_make_messy())
    p = tmp_path / "clean.parquet"
    out.to_parquet(p)
    pd.testing.assert_frame_equal(out, pd.read_parquet(p))


# ---------------------------------------------------------------------------
# Hardened tests on large seeded frame
# ---------------------------------------------------------------------------

def test_large_no_dupes():
    df = _make_large_messy()
    out = build_pipeline(df)
    assert out.duplicated().sum() == 0


def test_large_no_nans():
    df = _make_large_messy()
    out = build_pipeline(df)
    num_cols = out.select_dtypes(include="number").columns
    assert out[num_cols].isna().sum().sum() == 0


def test_large_group_normalized():
    df = _make_large_messy()
    out = build_pipeline(df)
    assert set(out["group"]) <= {"A", "B"}, f"Unexpected group values: {set(out['group'])}"


def test_large_row_count_equals_dedup():
    """Row count must equal the number of unique rows in the input."""
    df = _make_large_messy()
    expected_rows = len(df.drop_duplicates())
    out = build_pipeline(df)
    assert len(out) == expected_rows, (
        f"Expected {expected_rows} rows after dedup, got {len(out)}"
    )


def test_large_index_reset():
    """Index must be a contiguous 0-based RangeIndex after pipeline."""
    df = _make_large_messy()
    out = build_pipeline(df)
    assert list(out.index) == list(range(len(out))), "Index must be reset to 0..n-1"


def test_large_median_fill_x1():
    """x1 NaN values must be filled with the post-dedup median (not mean)."""
    df = _make_large_messy()
    deduped = df.drop_duplicates()
    expected_median = float(np.nanmedian(deduped["x1"].dropna().values))
    out = build_pipeline(df)
    assert out["x1"].isna().sum() == 0
    original_vals = set(deduped["x1"].dropna().values)
    for v in out["x1"]:
        assert v in original_vals or np.isclose(v, expected_median, atol=1e-6), (
            f"Unexpected x1 fill value {v} (expected median {expected_median})"
        )


def test_large_median_fill_x2():
    """x2 NaN values must be filled with the post-dedup median (not mean)."""
    df = _make_large_messy()
    deduped = df.drop_duplicates()
    expected_median = float(np.nanmedian(deduped["x2"].dropna().values))
    expected_mean = float(deduped["x2"].mean())
    out = build_pipeline(df)
    assert out["x2"].isna().sum() == 0
    if not np.isclose(expected_median, expected_mean, rtol=1e-3):
        original_non_nan = set(deduped["x2"].dropna().values)
        for v in out["x2"]:
            if v not in original_non_nan:
                assert np.isclose(v, expected_median, atol=1e-6), (
                    f"x2 fill value {v} ≠ median {expected_median}"
                )


def test_median_not_mean_trap():
    """Explicit trap: skewed numeric col — median ≠ mean; fill must be median."""
    df = pd.DataFrame({
        "age":   [10.0, 10.0, 10.0, 10.0, 1000.0, np.nan],
        "group": ["a", "b", "a", "b", "a", "b"],
        "x1":    [20.0] * 6,
        "x2":    [120.0] * 6,
        "x3":    [90.0] * 6,
        "label": [0, 1, 0, 1, 0, 1],
    })
    out = build_pipeline(df)
    assert out["age"].isna().sum() == 0
    assert 10.0 in out["age"].values, "Median fill (10.0) not found in age column"
    assert not any(
        abs(v - 208.0) < 5.0 and v not in {10.0, 1000.0}
        for v in out["age"].values
    ), "Mean fill detected in age column — should be median"


def test_required_columns_present():
    """All REQUIRED columns must survive the pipeline."""
    df = _make_large_messy()
    out = build_pipeline(df)
    for col in REQUIRED:
        assert col in out.columns, f"Required column '{col}' missing from output"
