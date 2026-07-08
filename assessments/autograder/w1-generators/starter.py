from __future__ import annotations
from typing import Iterable, Iterator


def moving_sum(xs: Iterable[float], k: int) -> Iterator[float]:
    """计算移动窗口和的生成器"""
    raise NotImplementedError


def take(it: Iterator, n: int) -> list:
    """从迭代器中取前 n 个元素"""
    raise NotImplementedError
