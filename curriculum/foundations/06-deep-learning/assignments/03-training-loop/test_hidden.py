import torch
from solution import train_linear


def test_learns_line():
    model, final_loss = train_linear()
    assert final_loss < 1e-3
    w = model.weight.item()
    b = model.bias.item()
    assert abs(w - 2.0) < 0.1 and abs(b - 1.0) < 0.1
