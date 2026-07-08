import numpy as np
import torch
import torch.nn.functional as F
from solution import Tensor, Linear, SGD, cross_entropy


def test_ce_matches_torch():
    rng = np.random.default_rng(0)
    logits_ = rng.normal(size=(6, 4)); y_ = rng.integers(0, 4, size=6)
    L = Tensor(logits_); cross_entropy(L, y_).backward()
    tL = torch.tensor(logits_, requires_grad=True)
    tloss = F.cross_entropy(tL, torch.tensor(y_))
    tloss.backward()
    assert np.isclose(cross_entropy(Tensor(logits_), y_).data, tloss.item())
    assert np.allclose(L.grad, tL.grad.numpy())


def test_classifier_reduces_ce():
    rng = np.random.default_rng(1)
    X_ = np.vstack([rng.normal(-1.5, 0.5, (30, 2)), rng.normal(1.5, 0.5, (30, 2))])
    y_ = np.array([0] * 30 + [1] * 30)
    lin = Linear(2, 2, seed=3); opt = SGD(lin.parameters(), lr=0.1)

    def ce():
        return cross_entropy(lin(Tensor(X_)), y_)
    l0 = ce().data
    for _ in range(300):
        opt.zero_grad(); l = ce(); l.backward(); opt.step()
    assert l.data < l0 * 0.5
