import numpy as np
from solution import kfold_indices


def test_partition():
    folds = kfold_indices(23, 5, seed=0)
    assert len(folds) == 5
    all_test = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(all_test), np.arange(23))     # every index tested exactly once
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0                 # no overlap
        assert len(tr) + len(te) == 23
