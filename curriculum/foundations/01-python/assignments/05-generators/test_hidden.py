import inspect
import itertools
import random
from solution import moving_sum, take


# ── generator identity ─────────────────────────────────────────────────────
def test_is_real_generator():
    g = moving_sum([1, 2, 3], 2)
    assert inspect.isgenerator(g), "moving_sum must return a generator, not a list"


def test_is_lazy_on_infinite_stream():
    # itertools.count() is infinite; a non-lazy impl would hang
    inf = itertools.count(0)
    g = moving_sum(inf, 3)
    assert inspect.isgenerator(g)
    first = next(g)   # must not hang; consume exactly one window
    assert first == 0 + 1 + 2  # 0,1,2 → 3


def test_take_from_infinite():
    result = take(moving_sum(itertools.count(0), 2), 5)
    # windows of 2 over 0,1,2,3,4,5: sums = 1,3,5,7,9
    assert result == [1, 3, 5, 7, 9]


# ── correctness on known inputs ────────────────────────────────────────────
def test_moving_sum_k2():
    assert list(moving_sum([1, 2, 3, 4], 2)) == [3, 5, 7]


def test_moving_sum_k3():
    assert list(moving_sum([1, 2, 3, 4, 5], 3)) == [6, 9, 12]


def test_moving_sum_k1_is_identity():
    # k=1: every element is its own window
    xs = [3, 1, 4, 1, 5, 9, 2, 6]
    assert list(moving_sum(xs, 1)) == xs


def test_moving_sum_k_equals_len():
    # k == len: exactly one output, the total sum
    xs = [2, 4, 6, 8]
    assert list(moving_sum(xs, 4)) == [20]


def test_moving_sum_k_larger_than_len():
    # no complete windows → empty output
    assert list(moving_sum([1, 2], 5)) == []


# ── random seeded correctness vs independent reference ─────────────────────
def _ref_moving_sum(xs, k):
    """Independent list-based reference (no generator)."""
    lst = list(xs)
    return [sum(lst[i:i + k]) for i in range(len(lst) - k + 1)]


def test_random_list_k2():
    rng = random.Random(101)
    xs = [rng.randint(1, 50) for _ in range(20)]
    assert list(moving_sum(xs, 2)) == _ref_moving_sum(xs, 2)


def test_random_list_k5():
    rng = random.Random(202)
    xs = [rng.uniform(-10, 10) for _ in range(15)]
    got = list(moving_sum(xs, 5))
    ref = _ref_moving_sum(xs, 5)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert abs(g - r) < 1e-9, f"got {g}, expected {r}"


def test_random_list_k_varies():
    rng = random.Random(303)
    for k in [1, 3, 7]:
        xs = [rng.randint(0, 100) for _ in range(20)]
        assert list(moving_sum(xs, k)) == _ref_moving_sum(xs, k), \
            f"mismatch at k={k}"


# ── take ───────────────────────────────────────────────────────────────────
def test_take_basic():
    assert take(moving_sum(range(100), 2), 3) == [1, 3, 5]


def test_take_zero():
    assert take(moving_sum(range(100), 2), 0) == []


def test_take_more_than_available():
    # request 10 but only 3 windows exist
    assert take(moving_sum([1, 2, 3, 4], 2), 10) == [3, 5, 7]
