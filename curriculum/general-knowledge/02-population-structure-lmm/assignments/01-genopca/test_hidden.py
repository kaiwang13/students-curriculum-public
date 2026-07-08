import numpy as np
from solution import genotype_pca


def test_pca_separates_populations():
    rng = np.random.default_rng(0)
    # two populations with different allele frequencies
    p1 = rng.uniform(0.1, 0.5, 50); p2 = rng.uniform(0.5, 0.9, 50)
    G1 = rng.binomial(2, p1, size=(100, 50))
    G2 = rng.binomial(2, p2, size=(100, 50))
    G = np.vstack([G1, G2]).astype(float)
    pcs = genotype_pca(G, k=2)
    assert pcs.shape == (200, 2)
    # PC1 separates the two groups
    assert abs(pcs[:100, 0].mean() - pcs[100:, 0].mean()) > 1.0
