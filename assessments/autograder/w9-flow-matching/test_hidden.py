import torch
from solution import ot_interpolate, velocity_target


def test_interpolate_endpoints():
    torch.manual_seed(0)
    x0 = torch.randn(4, 3); x1 = torch.randn(4, 3)
    assert torch.allclose(ot_interpolate(x0, x1, 0.0), x0)
    assert torch.allclose(ot_interpolate(x0, x1, 1.0), x1)
    assert torch.allclose(ot_interpolate(x0, x1, 0.5), 0.5 * (x0 + x1))


def test_velocity_is_displacement():
    x0 = torch.zeros(2, 3); x1 = torch.ones(2, 3)
    assert torch.allclose(velocity_target(x0, x1), torch.ones(2, 3))
    # integrating v over [0,1] from x0 recovers x1
    v = velocity_target(x0, x1)
    assert torch.allclose(x0 + 1.0 * v, x1)
