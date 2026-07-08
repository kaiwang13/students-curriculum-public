from solution import seed_everything, save_run, load_run


def test_determinism():
    a = seed_everything(0)
    b = seed_everything(0)
    c = seed_everything(1)
    assert a == b            # same seed -> identical draws
    assert a != c            # different seed -> different draws


def test_run_roundtrip(tmp_path):
    p = tmp_path / "run.json"
    cfg = {"lr": 0.01, "epochs": 10}
    met = {"auc": 0.83, "loss": 0.4}
    save_run(p, cfg, met)
    cfg2, met2 = load_run(p)
    assert cfg2 == cfg and met2 == met
