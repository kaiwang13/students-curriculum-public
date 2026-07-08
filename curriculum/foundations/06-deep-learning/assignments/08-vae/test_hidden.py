import torch
from solution import build_vae, train_vae, elbo_loss


def _data(n=128, d=6, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d)


def test_forward_shapes():
    model = build_vae(6, 2)
    X = _data()
    x_hat, mu, logvar = model(X)
    assert x_hat.shape == X.shape
    assert mu.shape == (128, 2) and logvar.shape == (128, 2)


def test_training_reduces_loss():
    X = _data()
    model = build_vae(X.shape[1], 2)
    x_hat, mu, logvar = model(X)
    loss0 = float(elbo_loss(x_hat, X, mu, logvar))
    _, final_loss = train_vae(X, epochs=300)
    assert final_loss < loss0        # ELBO loss decreases with training


def test_latent_dim():
    model = build_vae(6, 2)
    mu, logvar = model.encode(_data())
    assert mu.shape[1] == 2
