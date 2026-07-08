import numpy as np
import torch
from solution import Tensor, Linear, SGD


def test_linear_grad_matches_torch():
    rng = np.random.default_rng(0)
    X_ = rng.normal(size=(5, 3))
    lin = Linear(3, 2, seed=1)
    (lin(Tensor(X_))).sum().backward()
    tlin = torch.nn.Linear(3, 2).double()  # match float64 precision
    with torch.no_grad():
        tlin.weight.copy_(torch.tensor(lin.w.data.T))    # torch weight is (out, in)
        tlin.bias.copy_(torch.tensor(lin.b.data))
    tX = torch.tensor(X_, requires_grad=True)
    tlin(tX).sum().backward()
    assert np.allclose(lin.w.grad, tlin.weight.grad.numpy().T)
    assert np.allclose(lin.b.grad, tlin.bias.grad.numpy())


def test_sgd_reduces_loss():
    rng = np.random.default_rng(0)
    X_ = rng.normal(size=(64, 3)); w_true = np.array([1.0, -2.0, 0.5]); y_ = X_ @ w_true
    lin = Linear(3, 1, seed=2); opt = SGD(lin.parameters(), lr=0.05)

    def loss():
        diff = lin(Tensor(X_)) + Tensor(-y_.reshape(-1, 1))
        return (diff * diff).mean()
    l0 = loss().data
    for _ in range(300):
        opt.zero_grad(); l = loss(); l.backward(); opt.step()
    assert l.data < l0 * 0.05
