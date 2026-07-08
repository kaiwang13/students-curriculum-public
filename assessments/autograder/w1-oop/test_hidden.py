import math
import random
import pytest
from solution import Shape, Rectangle, Circle, total_area


# ── subclass / isinstance ──────────────────────────────────────────────────
def test_rectangle_is_shape():
    assert issubclass(Rectangle, Shape)
    assert isinstance(Rectangle(3, 4), Shape)


def test_circle_is_shape():
    assert issubclass(Circle, Shape)
    assert isinstance(Circle(5), Shape)


def test_shape_base_raises():
    with pytest.raises((NotImplementedError, TypeError)):
        Shape().area()


# ── Rectangle area = w * h ─────────────────────────────────────────────────
def test_rectangle_basic():
    assert Rectangle(2, 3).area() == 6
    assert Rectangle(1, 1).area() == 1
    assert Rectangle(7, 8).area() == 56


def test_rectangle_random_dims():
    rng = random.Random(42)
    for _ in range(8):
        w = rng.uniform(0.5, 20.0)
        h = rng.uniform(0.5, 20.0)
        assert math.isclose(Rectangle(w, h).area(), w * h), \
            f"Rectangle({w}, {h}).area() should be {w * h}"


def test_rectangle_asymmetric():
    # w*h != h*w in storage, but area must equal both orderings
    assert math.isclose(Rectangle(3, 7).area(), 21)
    assert math.isclose(Rectangle(7, 3).area(), 21)
    # trap: area != w+h (use local w, h — not private attributes)
    w, h = 4, 5
    assert Rectangle(w, h).area() == w * h


def test_rectangle_degenerate():
    assert Rectangle(0, 5).area() == 0
    assert Rectangle(5, 0).area() == 0
    assert Rectangle(0, 0).area() == 0


# ── Circle area = π r² ────────────────────────────────────────────────────
def test_circle_basic():
    assert math.isclose(Circle(1).area(), math.pi)
    assert math.isclose(Circle(2).area(), math.pi * 4)
    assert math.isclose(Circle(3).area(), math.pi * 9)


def test_circle_random_radii():
    rng = random.Random(99)
    for _ in range(8):
        r = rng.uniform(0.1, 15.0)
        expected = math.pi * r * r
        assert math.isclose(Circle(r).area(), expected, rel_tol=1e-9), \
            f"Circle({r}).area() should be {expected}"


def test_circle_not_2pi_r():
    # trap: area must not be 2*pi*r (circumference)
    r = 5.0
    assert not math.isclose(Circle(r).area(), 2 * math.pi * r)


def test_circle_zero_radius():
    assert Circle(0).area() == 0


# ── total_area ─────────────────────────────────────────────────────────────
def test_total_basic():
    assert math.isclose(total_area([Rectangle(2, 3), Circle(1)]), 6 + math.pi)


def test_total_empty():
    assert total_area([]) == 0


def test_total_single_rectangle():
    rng = random.Random(55)
    w, h = rng.uniform(1, 10), rng.uniform(1, 10)
    assert math.isclose(total_area([Rectangle(w, h)]), w * h)


def test_total_single_circle():
    rng = random.Random(66)
    r = rng.uniform(1, 10)
    assert math.isclose(total_area([Circle(r)]), math.pi * r * r)


def test_total_random_mix():
    rng = random.Random(7)
    shapes, expected = [], 0.0
    for _ in range(12):
        if rng.random() < 0.5:
            w, h = rng.uniform(1, 10), rng.uniform(1, 10)
            shapes.append(Rectangle(w, h))
            expected += w * h
        else:
            r = rng.uniform(1, 10)
            shapes.append(Circle(r))
            expected += math.pi * r * r
    assert math.isclose(total_area(shapes), expected, rel_tol=1e-9)


def test_total_all_rectangles():
    rng = random.Random(13)
    dims = [(rng.uniform(1, 5), rng.uniform(1, 5)) for _ in range(6)]
    rects = [Rectangle(w, h) for w, h in dims]
    expected = sum(w * h for w, h in dims)
    assert math.isclose(total_area(rects), expected)


def test_total_all_circles():
    rng = random.Random(17)
    radii = [rng.uniform(0.5, 8) for _ in range(6)]
    circles = [Circle(r) for r in radii]
    expected = sum(math.pi * r ** 2 for r in radii)
    assert math.isclose(total_area(circles), expected)
