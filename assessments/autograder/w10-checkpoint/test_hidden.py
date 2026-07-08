import torch
import torch.nn as nn
from solution import save_checkpoint, load_checkpoint


def test_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = nn.Linear(4, 3); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    # take one step so optimizer has state
    loss = m(torch.randn(5, 4)).sum(); loss.backward(); opt.step()
    p = tmp_path / "ckpt.pt"
    save_checkpoint(m, opt, step=42, path=p)
    m2 = nn.Linear(4, 3); opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    step = load_checkpoint(m2, opt2, p)
    assert step == 42
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1, p2)          # weights restored exactly
