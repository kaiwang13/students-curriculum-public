import numpy as np
from solution import loco_residualize


def test_loco_excludes_target_chrom():
    rng = np.random.default_rng(0)
    G = rng.integers(0, 3, size=(100, 12)).astype(float)
    chrom = np.array([1] * 4 + [2] * 4 + [3] * 4)
    y = rng.normal(size=100)
    r1 = loco_residualize(G, y, chrom, target_chrom=2, alpha=1.0)
    # corrupting chrom-2 SNPs must NOT change the residual (they're left out)
    G2 = G.copy(); G2[:, chrom == 2] = rng.integers(0, 3, size=(100, 4))
    r2 = loco_residualize(G2, y, chrom, target_chrom=2, alpha=1.0)
    assert np.allclose(r1, r2)


def test_loco_reduces_variance():
    rng = np.random.default_rng(1)
    G = rng.integers(0, 3, size=(120, 12)).astype(float)
    chrom = np.repeat([1, 2, 3], 4)
    y = G[:, 0] * 0.4 + rng.normal(scale=0.5, size=120)   # signal on chr1
    r = loco_residualize(G, y, target_chrom=2, chrom=chrom, alpha=0.1)
    assert (r.var()) < (y.var())
