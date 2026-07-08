"""nanograd L2：更多算子（^, 负数, -, /, exp, log, relu, tanh）。

本文件是 L2 的学生入门框架，扩展 L1 Value 类。
学生需要实现新的方法：__pow__, __neg__, __sub__, __truediv__, exp, log, relu, tanh。
"""

import math


class Value:
    """一个标量值，支持自动微分和更多运算。

    学生实现须包括：
    - __pow__: 幂运算符
    - __neg__: 负数运算符
    - __sub__: 减法运算符
    - __truediv__: 除法运算符
    - exp: 指数函数
    - log: 自然对数
    - relu: ReLU 激活函数
    - tanh: 双曲正切激活函数
    """

    def __init__(self, data, _children=(), _op=""):
        """初始化一个 Value 对象。

        Args:
            data: 标量数值
            _children: 子节点（默认为空）
            _op: 操作类型（默认为空字符串）
        """
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        """幂运算符。"""
        raise NotImplementedError

    def __neg__(self):
        """负数运算符。"""
        raise NotImplementedError

    def __sub__(self, other):
        """减法运算符。"""
        raise NotImplementedError

    def __truediv__(self, other):
        """除法运算符。"""
        raise NotImplementedError

    def exp(self):
        """指数函数。"""
        raise NotImplementedError

    def log(self):
        """自然对数。"""
        raise NotImplementedError

    def relu(self):
        """ReLU 激活函数。"""
        raise NotImplementedError

    def tanh(self):
        """双曲正切激活函数。"""
        raise NotImplementedError

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
