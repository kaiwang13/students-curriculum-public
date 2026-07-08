import torch
import torch.nn as nn
from solution import accumulate_and_step


def _make_linear(in_f, out_f, seed):
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f)


def test_accum_equals_full_batch():
    """Original test: 2 microbatches of 4 must match 1 full-batch step."""
    torch.manual_seed(0)
    X = torch.randn(8, 3); Y = torch.randn(8, 1)
    loss_fn = nn.MSELoss()

    def make():
        torch.manual_seed(1)
        return nn.Linear(3, 1)

    m_full = make(); opt_full = torch.optim.SGD(m_full.parameters(), lr=0.1)
    accumulate_and_step(m_full, opt_full, [(X, Y)], loss_fn)
    m_acc = make(); opt_acc = torch.optim.SGD(m_acc.parameters(), lr=0.1)
    accumulate_and_step(m_acc, opt_acc, [(X[:4], Y[:4]), (X[4:], Y[4:])], loss_fn)
    for (_, p1), (_, p2) in zip(m_full.named_parameters(), m_acc.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_accum_four_microbatches():
    """4 equal microbatches of 2 must match the full-batch step.
    Trap: an impl that steps after each microbatch (instead of once at end) fails here."""
    torch.manual_seed(0)
    X = torch.randn(8, 3); Y = torch.randn(8, 1)
    loss_fn = nn.MSELoss()

    def make():
        torch.manual_seed(1)
        return nn.Linear(3, 1)

    m_full = make(); opt_full = torch.optim.SGD(m_full.parameters(), lr=0.1)
    accumulate_and_step(m_full, opt_full, [(X, Y)], loss_fn)
    m_acc = make(); opt_acc = torch.optim.SGD(m_acc.parameters(), lr=0.1)
    microbatches = [(X[i * 2:(i + 1) * 2], Y[i * 2:(i + 1) * 2]) for i in range(4)]
    accumulate_and_step(m_acc, opt_acc, microbatches, loss_fn)
    for (_, p1), (_, p2) in zip(m_full.named_parameters(), m_acc.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "4 microbatches must match full-batch step"


def test_accum_eight_microbatches():
    """8 microbatches of 1 sample each must match the full-batch step.
    Trap: an impl that forgets to normalize by n_microbatches gives 8× the gradient."""
    torch.manual_seed(0)
    X = torch.randn(8, 3); Y = torch.randn(8, 1)
    loss_fn = nn.MSELoss()

    def make():
        torch.manual_seed(1)
        return nn.Linear(3, 1)

    m_full = make(); opt_full = torch.optim.SGD(m_full.parameters(), lr=0.1)
    accumulate_and_step(m_full, opt_full, [(X, Y)], loss_fn)
    m_acc = make(); opt_acc = torch.optim.SGD(m_acc.parameters(), lr=0.1)
    microbatches = [(X[i:i + 1], Y[i:i + 1]) for i in range(8)]
    accumulate_and_step(m_acc, opt_acc, microbatches, loss_fn)
    for (_, p1), (_, p2) in zip(m_full.named_parameters(), m_acc.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-5), "8 microbatches must match full-batch step"


def test_accum_random_data_four_way_split():
    """Random data, different feature dim, 4-way equal split."""
    torch.manual_seed(100)
    X = torch.randn(12, 5); Y = torch.randn(12, 1)
    loss_fn = nn.MSELoss()

    def make():
        torch.manual_seed(200)
        return nn.Linear(5, 1)

    m_full = make(); opt_full = torch.optim.SGD(m_full.parameters(), lr=0.05)
    accumulate_and_step(m_full, opt_full, [(X, Y)], loss_fn)
    m_acc = make(); opt_acc = torch.optim.SGD(m_acc.parameters(), lr=0.05)
    microbatches = [(X[i * 3:(i + 1) * 3], Y[i * 3:(i + 1) * 3]) for i in range(4)]
    accumulate_and_step(m_acc, opt_acc, microbatches, loss_fn)
    for (_, p1), (_, p2) in zip(m_full.named_parameters(), m_acc.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-5)


def test_returned_loss_equals_full_batch_loss():
    """The returned total must equal the full-batch MSE computed before the step.
    Trap: an impl that returns a per-microbatch loss (not the total) fails here."""
    torch.manual_seed(0)
    X = torch.randn(8, 3); Y = torch.randn(8, 1)
    loss_fn = nn.MSELoss()
    torch.manual_seed(1)
    model = nn.Linear(3, 1)
    with torch.no_grad():
        expected_loss = loss_fn(model(X), Y).item()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    microbatches = [(X[:4], Y[:4]), (X[4:], Y[4:])]
    returned = accumulate_and_step(model, opt, microbatches, loss_fn)
    assert abs(returned - expected_loss) < 1e-5, \
        f"returned loss {returned:.6f} != full-batch MSE {expected_loss:.6f}"


def test_single_microbatch_matches_standard_step():
    """With 1 microbatch, result must match a standard zero_grad/backward/step."""
    torch.manual_seed(0)
    X = torch.randn(4, 2); Y = torch.randn(4, 1)
    loss_fn = nn.MSELoss()

    def make():
        torch.manual_seed(1)
        return nn.Linear(2, 1)

    # Reference: manual one-step training
    m_ref = make(); opt_ref = torch.optim.SGD(m_ref.parameters(), lr=0.1)
    opt_ref.zero_grad()
    loss_fn(m_ref(X), Y).backward()
    opt_ref.step()

    m_acc = make(); opt_acc = torch.optim.SGD(m_acc.parameters(), lr=0.1)
    accumulate_and_step(m_acc, opt_acc, [(X, Y)], loss_fn)
    for (_, p1), (_, p2) in zip(m_ref.named_parameters(), m_acc.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)
