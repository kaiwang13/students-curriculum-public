import torch
from solution import euler_step, integrate


def test_euler_step():
    x = torch.zeros(3); v = torch.ones(3)
    assert torch.allclose(euler_step(x, v, 0.5), torch.full((3,), 0.5))


def test_integrate_constant_field():
    # dx/dt = (target - x0) constant -> reaches target at t=1 exactly (linear ODE, Euler exact for constant v)
    x0 = torch.zeros(4); target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = integrate(x0, lambda x, t: target - x0, n_steps=20)
    assert torch.allclose(out, target, atol=1e-5)
