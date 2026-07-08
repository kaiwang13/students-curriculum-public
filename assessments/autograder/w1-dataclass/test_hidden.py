import math
import random
import pytest
from solution import Point


# ── field access ───────────────────────────────────────────────────────────
def test_fields_accessible():
    p = Point(3.0, 4.0)
    assert p.x == 3.0
    assert p.y == 4.0


# ── equality (dataclass __eq__) ────────────────────────────────────────────
def test_equality_same_coords():
    assert Point(1, 2) == Point(1, 2)
    assert Point(0, 0) == Point(0, 0)


def test_inequality_different_coords():
    assert Point(1, 2) != Point(2, 1)   # trap: order matters
    assert Point(1, 2) != Point(1, 3)
    assert Point(1, 2) != Point(0, 2)


# ── frozen / immutability ──────────────────────────────────────────────────
def test_frozen_x():
    p = Point(1, 2)
    with pytest.raises(Exception):
        p.x = 9


def test_frozen_y():
    p = Point(1, 2)
    with pytest.raises(Exception):
        p.y = 9


def test_cannot_add_new_attr():
    p = Point(1, 2)
    with pytest.raises(Exception):
        p.z = 0


# ── dist_to: definition = hypot(dx, dy) ───────────────────────────────────
def test_dist_to_known():
    assert Point(0, 0).dist_to(Point(3, 4)) == 5.0
    assert Point(0, 0).dist_to(Point(0, 0)) == 0.0
    assert math.isclose(Point(1, 1).dist_to(Point(2, 2)), math.sqrt(2))


def test_dist_to_self_is_zero():
    p = Point(3.5, -2.7)
    assert p.dist_to(p) == 0.0


def test_dist_to_symmetry():
    rng = random.Random(21)
    for _ in range(6):
        a = Point(rng.uniform(-10, 10), rng.uniform(-10, 10))
        b = Point(rng.uniform(-10, 10), rng.uniform(-10, 10))
        assert math.isclose(a.dist_to(b), b.dist_to(a))


def test_dist_to_random():
    rng = random.Random(42)
    for _ in range(8):
        x1, y1 = rng.uniform(-20, 20), rng.uniform(-20, 20)
        x2, y2 = rng.uniform(-20, 20), rng.uniform(-20, 20)
        expected = math.hypot(x1 - x2, y1 - y2)
        got = Point(x1, y1).dist_to(Point(x2, y2))
        assert math.isclose(got, expected, rel_tol=1e-9), \
            f"dist_to wrong for ({x1},{y1})->({x2},{y2})"


def test_dist_to_not_constant():
    # trap: constant-return impl fails on multiple different distances
    # distances: 1.0, 5.0, sqrt(8)=2.828...
    distances = {
        Point(0, 0).dist_to(Point(1, 0)),
        Point(0, 0).dist_to(Point(0, 5)),
        Point(0, 0).dist_to(Point(2, 2)),
    }
    assert len(distances) == 3, "dist_to must vary with inputs"


# ── norm: definition = hypot(x, y) ────────────────────────────────────────
def test_norm_known():
    assert math.isclose(Point(3, 4).norm, 5.0)
    assert Point(0, 0).norm == 0.0
    assert math.isclose(Point(1, 0).norm, 1.0)
    assert math.isclose(Point(0, 1).norm, 1.0)


def test_norm_random():
    rng = random.Random(77)
    for _ in range(8):
        x, y = rng.uniform(-15, 15), rng.uniform(-15, 15)
        expected = math.hypot(x, y)
        got = Point(x, y).norm
        assert math.isclose(got, expected, rel_tol=1e-9), \
            f"norm wrong for ({x},{y})"


def test_norm_not_constant():
    norms = {Point(1, 0).norm, Point(0, 2).norm, Point(3, 4).norm}
    assert len(norms) == 3, "norm must vary with inputs"


# ── norm == dist_to origin ────────────────────────────────────────────────
def test_norm_equals_dist_to_origin():
    origin = Point(0, 0)
    rng = random.Random(55)
    for _ in range(6):
        x, y = rng.uniform(-10, 10), rng.uniform(-10, 10)
        p = Point(x, y)
        assert math.isclose(p.norm, p.dist_to(origin))
