import math
from solution import Value


def _grad(fn, x, h=1e-5):
    """Return (analytic_grad, numeric_central_diff) for fn at scalar x."""
    a = Value(x)
    fn(a).backward()
    analytic = a.grad
    numeric = (fn(Value(x + h)).data - fn(Value(x - h)).data) / (2 * h)
    return analytic, numeric


def _check(fn, x, tol=1e-4):
    """Assert analytic grad matches finite-difference oracle."""
    g, n = _grad(fn, x)
    assert abs(g - n) < tol, f"fn at x={x}: analytic={g:.8f}, numeric={n:.8f}"


def test_pow():
    g, n = _grad(lambda a: a ** 3, 2.0)
    assert abs(g - 12.0) < 1e-4 and abs(g - n) < 1e-4


def test_exp_log_tanh_relu():
    for fn, x in [(lambda a: a.exp(), 0.5), (lambda a: a.log(), 2.0),
                  (lambda a: a.tanh(), 0.3), (lambda a: a.relu(), 1.5),
                  (lambda a: a.relu(), -1.5)]:
        g, n = _grad(fn, x)
        assert abs(g - n) < 1e-4


def test_div_sub():
    a, b = Value(6.0), Value(3.0)
    (a / b).backward()
    assert abs(a.grad - 1 / 3) < 1e-9 and abs(b.grad - (-6.0 / 9)) < 1e-9


def test_composed_ops_fd_oracle():
    """Finite-difference oracle on several composed scalar expressions.
    Catches impls that break on multi-level computation graphs."""
    cases = [
        (lambda a: (a ** 2 + a).exp(),            0.3),
        (lambda a: (a * a + a + 1).log(),          0.7),
        (lambda a: a.tanh() * a + a,              1.2),
        (lambda a: (a ** 3 - a * 2).relu(),        1.5),
        (lambda a: (a.exp() + (-1)) * a,           0.1),
        (lambda a: a.log() + a ** 2,               1.5),
        (lambda a: (a ** 2).tanh() + a.exp(),      0.4),
        (lambda a: (a + 1) / (a ** 2 + 1),        1.0),
    ]
    for fn, x in cases:
        _check(fn, x)


def test_chain_of_ops():
    """Deeply composed chain: tanh(exp(x) + x^2)."""
    fn = lambda a: (a.exp() + a ** 2).tanh()
    _check(fn, 0.5)
    _check(fn, 1.0)
    _check(fn, 0.2)


def test_neg_and_sub():
    """Negation and subtraction."""
    _check(lambda a: -a + a ** 2, 1.5)
    _check(lambda a: a - a ** 2 * 3, 0.5)
    _check(lambda a: (-a).exp(), 0.8)


def test_div_composed():
    """Division inside a compound expression."""
    _check(lambda a: (a ** 2 + 1) / (a + 2), 1.0)
    _check(lambda a: a.exp() / (a ** 2 + 1),   0.8)
    _check(lambda a: a / (a.exp() + 1),         0.6)


def test_gradient_accumulation_two_paths():
    """A value used in two branches: grad must accumulate via both paths.
    f(a) = a*a + a  ->  f'(a) = 2a+1;   f(a) = a * exp(a) -> f'(a) = (1+a)*exp(a)."""
    _check(lambda a: a ** 2 + a,  3.0)
    _check(lambda a: a * a.exp(), 1.0)
    _check(lambda a: a.log() * a, 2.0)   # (ln a)*a -> 1 + ln a


def test_various_single_ops():
    """Each primitive at several random-ish but fixed inputs — different from the basic test."""
    cases = [
        (lambda a: a ** 4,    1.5),
        (lambda a: a ** 0.5,  4.0),
        (lambda a: a.exp(),   1.0),
        (lambda a: a.log(),   0.5),
        (lambda a: a.tanh(),  2.0),
        (lambda a: a.relu(),  0.01),
        (lambda a: a.relu(), -0.01),
    ]
    for fn, x in cases:
        _check(fn, x)
