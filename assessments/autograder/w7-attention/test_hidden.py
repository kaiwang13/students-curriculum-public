import torch
import torch.nn.functional as F
from solution import scaled_dot_product_attention


def test_matches_torch_sdpa():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    out, attn = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 5), atol=1e-5)   # rows are a distribution


def test_causal_mask():
    torch.manual_seed(1)
    q = k = v = torch.randn(1, 4, 8)
    mask = torch.tril(torch.ones(4, 4))
    _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.allclose(attn[0].triu(1), torch.zeros(4, 4), atol=1e-6)  # no attention to the future
