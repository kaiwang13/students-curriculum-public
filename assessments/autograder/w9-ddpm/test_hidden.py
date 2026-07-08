import torch
from solution import linear_beta_schedule, q_sample


def test_schedule():
    betas, alphas, alpha_bars = linear_beta_schedule(10)
    assert betas.shape == (10,)
    assert torch.all(alpha_bars[1:] <= alpha_bars[:-1])   # alpha_bar decreasing
    assert torch.allclose(alphas, 1 - betas)


def test_q_sample_endpoints():
    torch.manual_seed(0)
    _, _, ab = linear_beta_schedule(100)
    x0 = torch.randn(5, 3); noise = torch.randn(5, 3)
    xt0 = q_sample(x0, torch.zeros(5, dtype=torch.long), noise, ab)
    assert torch.allclose(xt0, torch.sqrt(ab[0]) * x0 + torch.sqrt(1 - ab[0]) * noise)
    # large t -> mostly noise (alpha_bar small)
    xtT = q_sample(x0, torch.full((5,), 99, dtype=torch.long), noise, ab)
    assert (xtT - noise).abs().mean() < (xtT - x0).abs().mean()
