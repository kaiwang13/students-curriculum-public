import functools
from solution import memoize, tag


# ── memoize: caching behaviour ─────────────────────────────────────────────
def test_memoize_caches_single_arg():
    calls = []

    @memoize
    def fn(x):
        calls.append(x)
        return x * x

    assert fn(3) == 9
    assert fn(3) == 9
    assert len(calls) == 1   # called exactly once for the same arg


def test_memoize_different_args_each_computed():
    calls = []

    @memoize
    def fn(x):
        calls.append(x)
        return x + 10

    assert fn(1) == 11
    assert fn(2) == 12
    assert fn(1) == 11   # cached
    assert calls == [1, 2]  # 2 distinct calls, not 3


def test_memoize_return_value_preserved():
    @memoize
    def fn(x):
        return x * 3

    # return value must match the computation, not a constant
    for v in [0, 1, 5, 7, -4, 100]:
        assert fn(v) == v * 3


def test_memoize_multiple_args():
    calls = []

    @memoize
    def add(a, b):
        calls.append((a, b))
        return a + b

    assert add(2, 3) == 5
    assert add(2, 3) == 5
    assert add(3, 2) == 5        # different arg order → separate cache entry
    assert calls.count((2, 3)) == 1
    assert calls.count((3, 2)) == 1


def test_memoize_independent_functions():
    # each decorated function has its own cache
    @memoize
    def double(x):
        return x * 2

    @memoize
    def triple(x):
        return x * 3

    for v in [1, 2, 3, 10]:
        assert double(v) == v * 2
        assert triple(v) == v * 3


def test_memoize_preserves_return_type():
    @memoize
    def fn(x):
        return [x, x + 1]

    assert fn(4) == [4, 5]
    assert fn(4) == [4, 5]   # cached, same result


def test_memoize_wraps_preserves_name():
    @memoize
    def my_special_fn(x):
        return x

    assert my_special_fn.__name__ == "my_special_fn"


# ── tag context manager ────────────────────────────────────────────────────
def test_tag_basic():
    log = []
    with tag("a", log):
        log.append("body")
    assert log == ["start:a", "body", "end:a"]


def test_tag_different_names():
    for name in ["foo", "bar", "setup", "teardown"]:
        log = []
        with tag(name, log):
            pass
        assert log == [f"start:{name}", f"end:{name}"], \
            f"wrong log for tag name '{name}'"


def test_tag_nested():
    log = []
    with tag("outer", log):
        with tag("inner", log):
            log.append("work")
    assert log == ["start:outer", "start:inner", "work", "end:inner", "end:outer"]


def test_tag_end_logged_on_exception():
    log = []
    with tag("cleanup", log):
        try:
            raise ValueError("oops")
        except ValueError:
            pass
    # end must be logged even if exception was raised inside
    assert log[0] == "start:cleanup"
    assert log[-1] == "end:cleanup"


def test_tag_exception_propagates():
    log = []
    try:
        with tag("x", log):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # end must still be logged when exception escapes
    assert "start:x" in log
    assert "end:x" in log


def test_tag_body_runs_between_markers():
    log = []
    with tag("t", log):
        log.append("mid")
    assert log.index("start:t") < log.index("mid") < log.index("end:t")
