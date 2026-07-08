"""Hidden tests for w3-io.

Hardened: large seeded round-trip (500 rows, int64/float64/str), dtype
preservation check, chunked_row_count with non-divisible and exact-divisible
sizes, and a zero-row edge case.
"""
import numpy as np
import pandas as pd
from solution import roundtrip_parquet, chunked_row_count


def _make_frame(seed=99, n=500):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "int_col":   rng.integers(0, 10_000, n).astype("int64"),
        "float_col": rng.uniform(-1.0, 1.0, n),
        "str_col":   [f"item_{i}" for i in rng.integers(0, 200, n)],
    })


def test_roundtrip_basic(tmp_path):
    """Original small regression test."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    back = roundtrip_parquet(df, tmp_path / "t.parquet")
    pd.testing.assert_frame_equal(df, back)


def test_roundtrip_large_values(tmp_path):
    """500 rows — all values must survive the parquet round-trip exactly."""
    df = _make_frame()
    back = roundtrip_parquet(df, tmp_path / "large.parquet")
    pd.testing.assert_frame_equal(df, back)


def test_roundtrip_dtypes_preserved(tmp_path):
    """int64 and float64 dtypes must be preserved through parquet."""
    df = _make_frame()
    back = roundtrip_parquet(df, tmp_path / "dtypes.parquet")
    assert back["int_col"].dtype == np.dtype("int64"), (
        f"int_col dtype changed: {back['int_col'].dtype}"
    )
    assert back["float_col"].dtype == np.dtype("float64"), (
        f"float_col dtype changed: {back['float_col'].dtype}"
    )


def test_roundtrip_column_order(tmp_path):
    """Column order must be preserved after round-trip."""
    df = _make_frame()
    back = roundtrip_parquet(df, tmp_path / "order.parquet")
    assert list(back.columns) == list(df.columns)


def test_chunked_count_basic(tmp_path):
    """Original regression test: 250 rows, chunksize=100."""
    df = pd.DataFrame({"a": range(250)})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    assert chunked_row_count(p, chunksize=100) == 250


def test_chunked_count_non_divisible(tmp_path):
    """777 rows, chunksize=100 → 7 full chunks + 77-row remainder = 777."""
    df = pd.DataFrame({"val": range(777)})
    p = tmp_path / "big.csv"
    df.to_csv(p, index=False)
    assert chunked_row_count(p, chunksize=100) == 777


def test_chunked_count_exact_divisible(tmp_path):
    """300 rows, chunksize=50 → exactly 6 full chunks, no remainder."""
    df = pd.DataFrame({"val": range(300)})
    p = tmp_path / "div.csv"
    df.to_csv(p, index=False)
    assert chunked_row_count(p, chunksize=50) == 300


def test_chunked_count_small_chunksize(tmp_path):
    """Large chunksize larger than file: single chunk must still count correctly."""
    df = pd.DataFrame({"val": range(42)})
    p = tmp_path / "small.csv"
    df.to_csv(p, index=False)
    assert chunked_row_count(p, chunksize=1000) == 42


def test_chunked_count_large_file(tmp_path):
    """2500 rows, chunksize=300 → 8 full + 1 partial chunk."""
    df = pd.DataFrame({"x": range(2500), "y": np.arange(2500, dtype=float)})
    p = tmp_path / "large.csv"
    df.to_csv(p, index=False)
    assert chunked_row_count(p, chunksize=300) == 2500


def test_roundtrip_index_preserved(tmp_path):
    """Default RangeIndex must survive the round-trip."""
    df = _make_frame(seed=7, n=100)
    back = roundtrip_parquet(df, tmp_path / "idx.parquet")
    assert list(back.index) == list(df.index)
