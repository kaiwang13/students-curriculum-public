import numpy as np
import torch
from solution import Tensor


def test_matmul_add_relu_grads_match_torch():
    rng = np.random.default_rng(0)
    A_ = rng.normal(size=(3, 4)); B_ = rng.normal(size=(4, 2)); c_ = rng.normal(size=(2,))
    A, B, c = Tensor(A_), Tensor(B_), Tensor(c_)
    ((A @ B) + c).relu().sum().backward()
    tA = torch.tensor(A_, requires_grad=True)
    tB = torch.tensor(B_, requires_grad=True)
    tc = torch.tensor(c_, requires_grad=True)
    ((tA @ tB) + tc).relu().sum().backward()
    assert np.allclose(A.grad, tA.grad.numpy())
    assert np.allclose(B.grad, tB.grad.numpy())
    assert np.allclose(c.grad, tc.grad.numpy())   # broadcasting backward for the bias


def test_elementwise_broadcast():
    x_ = np.random.default_rng(1).normal(size=(5, 3))
    x = Tensor(x_); (x * x).sum().backward()
    tx = torch.tensor(x_, requires_grad=True); (tx * tx).sum().backward()
    assert np.allclose(x.grad, tx.grad.numpy())
