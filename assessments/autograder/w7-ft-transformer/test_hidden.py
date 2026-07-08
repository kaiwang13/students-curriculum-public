import torch
from solution import FeatureTokenizer


def test_tokenizer_shape():
    torch.manual_seed(0)
    tok = FeatureTokenizer(n_features=6, d_token=8)
    out = tok(torch.randn(4, 6))
    assert out.shape == (4, 7, 8)                 # +1 for the CLS token


def test_cls_prepended():
    torch.manual_seed(0)
    tok = FeatureTokenizer(3, 5)
    out = tok(torch.zeros(2, 3))                  # x=0 -> tokens == bias == 0; CLS is the learned param
    assert torch.allclose(out[:, 0, :], tok.cls.expand(2, 5))
    assert torch.allclose(out[:, 1:, :], torch.zeros(2, 3, 5))
