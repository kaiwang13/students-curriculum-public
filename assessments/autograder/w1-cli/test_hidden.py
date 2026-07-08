import pytest
from solution import parse


# ── defaults ───────────────────────────────────────────────────────────────
def test_defaults():
    r = parse([])
    assert r == {"n": 1, "name": "world", "verbose": False}


def test_default_n_is_int():
    # trap: a constant-string impl would return str not int
    assert isinstance(parse([])["n"], int)


def test_default_verbose_is_bool_false():
    assert parse([])["verbose"] is False


# ── individual flags ───────────────────────────────────────────────────────
def test_n_only():
    r = parse(["--n", "7"])
    assert r["n"] == 7
    assert r["name"] == "world"   # other defaults unchanged
    assert r["verbose"] is False


def test_name_only():
    r = parse(["--name", "alice"])
    assert r["name"] == "alice"
    assert r["n"] == 1
    assert r["verbose"] is False


def test_verbose_flag_sets_true():
    r = parse(["--verbose"])
    assert r["verbose"] is True
    assert r["n"] == 1
    assert r["name"] == "world"


# ── type coercion ──────────────────────────────────────────────────────────
def test_n_coerced_to_int():
    for val in ["3", "10", "0", "100"]:
        result = parse(["--n", val])
        assert isinstance(result["n"], int), "--n must be an int"
        assert result["n"] == int(val)


def test_name_stays_string():
    for name in ["bob", "Carol", "X Æ A-12", "123"]:
        assert isinstance(parse(["--name", name])["name"], str)
        assert parse(["--name", name])["name"] == name


# ── combined args ──────────────────────────────────────────────────────────
def test_all_args():
    r = parse(["--n", "5", "--name", "amy", "--verbose"])
    assert r == {"n": 5, "name": "amy", "verbose": True}


def test_n_and_name_no_verbose():
    r = parse(["--n", "42", "--name", "zhang"])
    assert r == {"n": 42, "name": "zhang", "verbose": False}


def test_n_and_verbose_no_name():
    r = parse(["--n", "99", "--verbose"])
    assert r == {"n": 99, "name": "world", "verbose": True}


# ── return type is a plain dict ────────────────────────────────────────────
def test_returns_dict():
    r = parse([])
    assert isinstance(r, dict)
    assert set(r.keys()) == {"n", "name", "verbose"}


# ── trap: multiple different values must parse independently ───────────────
def test_different_n_values():
    vals = [2, 10, 0, 999, 1]
    for v in vals:
        r = parse(["--n", str(v)])
        assert r["n"] == v, f"Expected n={v}, got {r['n']}"
