import torch
from solution import MLP


def test_forward_shape():
    torch.manual_seed(0)
    m = MLP(4, 8, 2)
    out = m(torch.randn(5, 4))
    assert out.shape == (5, 2)
    assert sum(p.numel() for p in m.parameters()) == 4 * 8 + 8 + 8 * 2 + 2
