from __future__ import annotations
from functools import wraps
from contextlib import contextmanager


def memoize(fn):
    """装饰器，缓存函数的计算结果"""
    raise NotImplementedError


@contextmanager
def tag(name: str, log: list):
    """上下文管理器，记录代码块的开始和结束"""
    raise NotImplementedError
