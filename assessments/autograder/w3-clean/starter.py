import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗：去重、数值列用中位数填补缺失、把 group 统一为大写。返回新 DataFrame。"""
    raise NotImplementedError
