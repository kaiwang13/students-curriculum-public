import numpy as np
from solution import maf, mac


def test_maf_mac_basic():
    """Original structured test — hand-computed reference values."""
    G = np.array([[0, 2], [1, 2], [2, 2], [0, 0]])   # snp0 ac=3 of 8; snp1 ac=6 of 8
    assert np.allclose(maf(G), [3 / 8, 2 / 8])
    assert np.allclose(mac(G), [3, 2])


def test_maf_range():
    """Property: maf values must lie in [0, 0.5] for all SNPs on random genotype matrices."""
    rng = np.random.default_rng(42)
    G = rng.integers(0, 3, size=(50, 200))
    m = maf(G)
    assert m.shape == (200,)
    assert np.all(m >= 0.0) and np.all(m <= 0.5)


def test_maf_flip_invariance():
    """Property: maf(G) == maf(2-G) — flipping allele codes preserves minor allele frequency."""
    rng = np.random.default_rng(7)
    G = rng.integers(0, 3, size=(30, 100))
    assert np.allclose(maf(G), maf(2 - G))


def test_mac_equals_maf_times_2n():
    """Property: mac == maf * 2 * n_samples exactly (both derive from integer allele counts)."""
    rng = np.random.default_rng(13)
    G = rng.integers(0, 3, size=(40, 60))
    n = G.shape[0]
    assert np.allclose(mac(G), maf(G) * 2 * n)


def test_monomorphic_columns_maf_zero():
    """Edge case: all-0 column and all-2 column are monomorphic → maf=0, mac=0."""
    G = np.array([
        [0, 2, 1],
        [0, 2, 0],
        [0, 2, 2],
    ])
    m = maf(G)
    mc = mac(G)
    # Column 0: all 0s → minor allele count 0, maf 0
    assert m[0] == 0.0 and mc[0] == 0.0
    # Column 1: all 2s → minor allele count 0, maf 0
    assert m[1] == 0.0 and mc[1] == 0.0


def test_maf_hand_computed_additional():
    """Additional hand-computed values to pin exact behavior."""
    # col0: [2,2,0] → ac=4, n_total=6, p=4/6, maf=1-2/3=1/3; mac=min(4,2)=2
    # col1: [1,1,1] → ac=3, n_total=6, p=1/2, maf=1/2; mac=min(3,3)=3
    G = np.array([[2, 1], [2, 1], [0, 1]])
    assert np.allclose(maf(G), [1 / 3, 1 / 2])
    assert np.allclose(mac(G), [2, 3])


def test_trap_constant_impl_fails():
    """Trap: a constant-output impl returns wrong values for monomorphic columns.
    Any impl that returns a nonzero constant for all SNPs fails here."""
    G_zeros = np.zeros((10, 5))       # all monomorphic-0
    G_twos = np.full((10, 5), 2.0)   # all monomorphic-2
    assert np.allclose(maf(G_zeros), 0.0)
    assert np.allclose(mac(G_zeros), 0.0)
    assert np.allclose(maf(G_twos), 0.0)
    assert np.allclose(mac(G_twos), 0.0)


def test_maf_multiple_sizes():
    """Property checks across several random matrix shapes."""
    rng = np.random.default_rng(99)
    for n, m_snps in [(5, 3), (100, 10), (20, 50)]:
        G = rng.integers(0, 3, size=(n, m_snps))
        mv = maf(G)
        mcv = mac(G)
        assert mv.shape == (m_snps,)
        assert mcv.shape == (m_snps,)
        assert np.all(mv >= 0.0) and np.all(mv <= 0.5)
        assert np.allclose(mcv, mv * 2 * n)
        assert np.allclose(mv, maf(2 - G))   # flip invariance at each size
