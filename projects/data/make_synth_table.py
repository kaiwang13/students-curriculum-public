"""种子化的合成表格数据生成器（离线、可复现），用于练习与项目。"""
from __future__ import annotations
import numpy as np
import pandas as pd

_COLS = ["record_id", "age", "group", "x1", "x2", "x3", "label"]


def make_synth_table(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """生成 n 行合成表格数据。label 为由若干特征线性组合生成的二分类标签。"""
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 90, n)
    group = rng.choice(["A", "B"], n)
    x1 = np.clip(rng.normal(27, 5, n), 14, 55)
    x2 = np.clip(rng.normal(125, 18, n) + 0.2 * (age - 50), 80, 220)
    x3 = np.clip(rng.normal(95, 20, n) + 0.15 * (x1 - 25) * 5, 55, 300)
    # 由若干特征线性组合生成的二分类标签
    score = 0.03 * (age - 50) + 0.05 * (x1 - 25) + 0.02 * (x3 - 95)
    prob = 1 / (1 + np.exp(-score))
    label = (rng.uniform(0, 1, n) < prob).astype(int)
    df = pd.DataFrame({
        "record_id": np.arange(n),
        "age": age, "group": group, "x1": x1, "x2": x2,
        "x3": x3, "label": label,
    })
    return df[_COLS]


def make_messy_table(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """生成含重复行、缺失值、大小写混合 group 的脏合成表格数据。"""
    rng = np.random.default_rng(seed)
    df = make_synth_table(n=n, seed=seed).copy()
    # mixed-case group strings
    df["group"] = [s.lower() if rng.uniform() < 0.5 else s for s in df["group"]]
    # inject NaNs into a few numeric columns
    for col in ("x1", "x3", "x2"):
        idx = rng.choice(n, size=max(1, n // 20), replace=False)
        df.loc[idx, col] = np.nan
    # duplicate a block of rows
    dupes = df.iloc[: max(2, n // 50)].copy()
    return pd.concat([df, dupes], ignore_index=True)
