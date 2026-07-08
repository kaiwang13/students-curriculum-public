import math


class Shape:
    def area(self) -> float:
        """计算形状的面积"""
        raise NotImplementedError


class Rectangle(Shape):
    def __init__(self, w: float, h: float):
        """初始化矩形，参数为宽度和高度"""
        raise NotImplementedError

    def area(self) -> float:
        """计算矩形的面积"""
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, r: float):
        """初始化圆形，参数为半径"""
        raise NotImplementedError

    def area(self) -> float:
        """计算圆形的面积"""
        raise NotImplementedError


def total_area(shapes: list[Shape]) -> float:
    """计算所有形状的总面积"""
    raise NotImplementedError
