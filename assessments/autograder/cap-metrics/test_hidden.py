import numpy as np
from solution import aggregate_runs


def test_aggregate():
    runs = [{"auc": 0.80}, {"auc": 0.84}, {"auc": 0.82}]
    out = aggregate_runs(runs)
    assert np.isclose(out["auc"]["mean"], 0.82)
    assert np.isclose(out["auc"]["std"], np.std([0.80, 0.84, 0.82], ddof=1))
    assert np.isclose(out["auc"]["ci95"], 1.96 * out["auc"]["std"] / np.sqrt(3))


def test_multi_metric():
    runs = [{"auc": 0.8, "loss": 0.5}, {"auc": 0.9, "loss": 0.3}]
    out = aggregate_runs(runs)
    assert set(out.keys()) == {"auc", "loss"}
    assert np.isclose(out["loss"]["mean"], 0.4)
