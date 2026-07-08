import torch
from solution import MultiHeadAttention


def test_shape_and_params():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 5, 16)
    out = mha(x)
    assert out.shape == (2, 5, 16)
    assert sum(p.numel() for p in mha.parameters()) == 4 * (16 * 16 + 16)  # 4 linear layers


def test_deterministic():
    torch.manual_seed(0); mha = MultiHeadAttention(8, 2); x = torch.randn(1, 3, 8)
    a = mha(x)
    b = mha(x)
    assert torch.allclose(a, b)
