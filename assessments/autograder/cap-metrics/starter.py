import numpy as np


def aggregate_runs(runs):
    """把多个随机种子的指标字典聚合为 {metric: {'mean','std','ci95'}}。
    std 用样本标准差 (ddof=1)；ci95 = 1.96 * std / sqrt(n) 的半宽（n=1 时为 0）。
    runs: list[dict[str, float]]，每个 dict 的键相同。"""
    raise NotImplementedError
