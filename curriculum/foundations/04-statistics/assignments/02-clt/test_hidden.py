import numpy as np
from solution import standard_error, sample_means


def test_standard_error():
    x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert np.isclose(standard_error(x), np.std(x, ddof=1) / np.sqrt(len(x)))


def test_sample_means_shape_and_center():
    data = np.arange(100.0)
    sm = sample_means(data, n=20, reps=500, seed=1)
    assert sm.shape == (500,)
    assert np.isclose(sm.mean(), data.mean(), atol=1.0)   # centered on the population mean
