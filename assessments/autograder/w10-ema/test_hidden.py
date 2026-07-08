import torch
import torch.nn as nn
from solution import EMA


def test_ema_formula():
    torch.manual_seed(0)
    m = nn.Linear(3, 2)
    ema = EMA(m, decay=0.9)
    init = {n: p.detach().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)                       # move params by +1
    ema.update(m)
    for n, p in m.named_parameters():
        expected = 0.9 * init[n] + 0.1 * (init[n] + 1.0)   # shadow update
        assert torch.allclose(ema.shadow[n], expected)


def test_copy_to():
    torch.manual_seed(0)
    m = nn.Linear(2, 2); ema = EMA(m, decay=0.5)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(0.0)                        # zero the live params
    ema.copy_to(m)                             # restore from shadow (the init values)
    for n, p in m.named_parameters():
        assert torch.allclose(p, ema.shadow[n])
