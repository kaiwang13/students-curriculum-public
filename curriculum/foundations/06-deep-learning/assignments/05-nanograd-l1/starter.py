"""nanograd L1：标量自动微分（+ 与 *），拓扑排序反向传播。

本文件是 L1 的学生入门框架。
学生需要实现 __add__, __mul__, 和 backward 方法。
"""


class Value:
    """一个标量值，支持自动微分。

    学生实现须包括：
    - __add__: 加法运算符
    - __mul__: 乘法运算符
    - backward: 反向传播
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
        """加法运算符。"""
        raise NotImplementedError

    def __mul__(self, other):
        """乘法运算符。"""
        raise NotImplementedError

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        """反向传播，计算所有参数的梯度。"""
        raise NotImplementedError
