import torch
from solution import reparameterize, kl_divergence


def test_reparameterize_shape_and_zero_var():
    mu = torch.zeros(4, 3)
    logvar = torch.full((4, 3), -1e9)     # std ~ 0 -> z ~ mu
    z = reparameterize(mu, logvar)
    assert z.shape == (4, 3)
    assert torch.allclose(z, mu, atol=1e-3)


def test_kl_zero_at_standard_normal():
    mu = torch.zeros(8, 5)
    logvar = torch.zeros(8, 5)            # posterior == prior -> KL == 0
    assert torch.allclose(kl_divergence(mu, logvar), torch.tensor(0.0), atol=1e-6)


def test_kl_positive_otherwise():
    mu = torch.ones(8, 5)
    logvar = torch.zeros(8, 5)
    assert kl_divergence(mu, logvar).item() > 0
