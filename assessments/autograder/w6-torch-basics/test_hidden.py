import torch
from solution import grad_of_square_sum


def test_grad_is_2x():
    """Original test."""
    g = grad_of_square_sum([1.0, 2.0, 3.0])
    assert torch.allclose(g, torch.tensor([2.0, 4.0, 6.0]))


def test_grad_random_input():
    """Property: grad == 2*x for a random 20-element input."""
    torch.manual_seed(42)
    x = torch.randn(20)
    g = grad_of_square_sum(x.tolist())
    assert torch.allclose(g, 2.0 * x, atol=1e-6), "grad must equal 2*x"


def test_grad_negative_values():
    """Works correctly for negative and mixed-sign inputs."""
    x = torch.tensor([-3.0, -1.0, 0.5, 2.0])
    g = grad_of_square_sum(x.tolist())
    assert torch.allclose(g, 2.0 * x, atol=1e-6)


def test_grad_zero_vector():
    """Gradient of sum(0^2) is all zeros."""
    g = grad_of_square_sum([0.0, 0.0, 0.0])
    assert torch.allclose(g, torch.zeros(3))


def test_grad_single_element():
    """Works for single-element input: grad([x]) == [2x]."""
    g = grad_of_square_sum([3.0])
    assert torch.allclose(g, torch.tensor([6.0]))


def test_output_shape():
    """Output shape matches input length."""
    g = grad_of_square_sum([1.0] * 7)
    assert g.shape == (7,)


def test_grad_multiple_sizes():
    """Verify grad == 2*x across several sizes with random data."""
    torch.manual_seed(7)
    for n in [1, 5, 50]:
        x = torch.randn(n)
        g = grad_of_square_sum(x.tolist())
        assert torch.allclose(g, 2.0 * x, atol=1e-6), f"Failed for n={n}"


def test_trap_hardcoded_output_fails():
    """Trap: a hardcoded or constant-output impl fails across different random inputs.
    grad at x must equal 2*x, so two different inputs give two different gradients."""
    torch.manual_seed(0)
    x1 = torch.randn(5)
    x2 = torch.randn(5)
    g1 = grad_of_square_sum(x1.tolist())
    g2 = grad_of_square_sum(x2.tolist())
    assert torch.allclose(g1, 2.0 * x1, atol=1e-6)
    assert torch.allclose(g2, 2.0 * x2, atol=1e-6)
    # The two gradient vectors should differ (x1 != x2 with overwhelming probability)
    assert not torch.allclose(g1, g2, atol=1e-6), \
        "Gradient must depend on input, not be constant"
