from __future__ import annotations
from pathlib import Path
import pandas as pd


def roundtrip_parquet(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """把 df 写成 parquet 再读回，返回读回的 DataFrame。"""
    raise NotImplementedError


def chunked_row_count(csv_path: str | Path, chunksize: int = 100) -> int:
    """用分块读取统计 CSV 的数据行数（不一次性载入）。"""
    raise NotImplementedError
