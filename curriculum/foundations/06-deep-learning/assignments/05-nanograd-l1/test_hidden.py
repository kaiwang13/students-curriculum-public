from solution import Value


def test_forward():
    a, b = Value(2.0), Value(3.0)
    assert (a * b + a).data == 8.0


def test_backward_add_mul():
    a, b = Value(2.0), Value(3.0)
    f = a * b + a          # f = a*b + a  -> df/da = b+1 = 4, df/db = a = 2
    f.backward()
    assert a.grad == 4.0
    assert b.grad == 2.0


def test_grad_accumulates():
    a = Value(3.0)
    f = a + a              # df/da = 2
    f.backward()
    assert a.grad == 2.0


def test_finite_difference():
    def fn(x, y):
        return x * y + x * x    # df/dx = y + 2x, df/dy = x
    a, b = Value(2.0), Value(-3.0)
    out = fn(a, b)
    out.backward()
    eps = 1e-6
    num_da = (fn(Value(2.0 + eps), Value(-3.0)).data - fn(Value(2.0 - eps), Value(-3.0)).data) / (2 * eps)
    num_db = (fn(Value(2.0), Value(-3.0 + eps)).data - fn(Value(2.0), Value(-3.0 - eps)).data) / (2 * eps)
    assert abs(a.grad - num_da) < 1e-4
    assert abs(b.grad - num_db) < 1e-4
