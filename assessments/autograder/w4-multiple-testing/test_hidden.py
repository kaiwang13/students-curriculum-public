import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from solution import bonferroni, bh_fdr


# ── original smoke tests ──────────────────────────────────────────────────────

def test_bonferroni():
    out = bonferroni([0.001, 0.04, 0.5], alpha=0.05)   # threshold 0.05/3 ≈ 0.0167
    assert out.tolist() == [True, False, False]


def test_bh_fdr():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    out = bh_fdr(p, alpha=0.05)                          # BH rejects the two smallest
    assert out.tolist() == [True, True, False, False, False]


# ── oracle tests: compare against statsmodels on randomised inputs ────────────

@pytest.mark.parametrize("seed,n", [(1, 20), (2, 50), (42, 15), (100, 30)])
def test_bonferroni_oracle_random(seed, n):
    """Bonferroni must agree with statsmodels on random p-value vectors."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 0.3, n)          # range gives a mix of rejections
    ref, _, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
    out = bonferroni(p, alpha=0.05)
    assert np.array_equal(out, ref), f"seed={seed} n={n}: got {out}, expected {ref}"


@pytest.mark.parametrize("seed,n", [(3, 20), (7, 50), (13, 10), (99, 40)])
def test_bh_fdr_oracle_random(seed, n):
    """BH FDR must agree with statsmodels on random p-value vectors."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 0.3, n)
    ref, _, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    out = bh_fdr(p, alpha=0.05)
    assert np.array_equal(out, ref), f"seed={seed} n={n}: got {out}, expected {ref}"


# ── edge cases ────────────────────────────────────────────────────────────────

def test_bonferroni_all_null():
    """All large p-values → no rejections."""
    rng = np.random.default_rng(5)
    p = rng.uniform(0.5, 1.0, 15)
    ref, _, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
    assert not ref.any()
    assert not bonferroni(p, alpha=0.05).any()


def test_bonferroni_all_significant():
    """All tiny p-values → all rejected."""
    rng = np.random.default_rng(6)
    p = rng.uniform(0.0, 0.0005, 15)
    ref, _, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
    assert ref.all()
    assert bonferroni(p, alpha=0.05).all()


def test_bh_fdr_all_null():
    """All large p-values → no rejections under BH."""
    rng = np.random.default_rng(8)
    p = rng.uniform(0.5, 1.0, 15)
    ref, _, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    assert not ref.any()
    assert not bh_fdr(p, alpha=0.05).any()


def test_bh_fdr_all_significant():
    """All tiny p-values → all rejected under BH."""
    rng = np.random.default_rng(9)
    p = rng.uniform(0.0, 0.001, 15)
    ref, _, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    assert ref.all()
    assert bh_fdr(p, alpha=0.05).all()


def test_bh_fdr_preserves_input_order():
    """BH result must be in the original (unsorted) index order, not rank order."""
    rng = np.random.default_rng(11)
    p = rng.uniform(0, 0.2, 25)
    ref, _, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    out = bh_fdr(p, alpha=0.05)
    assert np.array_equal(out, ref), "result order must match input order"


# ── trap: constant-return naive wrong implementations ─────────────────────────

def test_bonferroni_trap_not_constant_true():
    """A naive 'always reject' stub must fail."""
    p = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = bonferroni(p, alpha=0.05)
    assert not np.all(out), "threshold 0.05/5=0.01 → none should be rejected"


def test_bh_fdr_trap_not_constant_true():
    """A naive 'always reject' stub must fail on an all-large-p input."""
    p = [0.4, 0.5, 0.6, 0.7, 0.8]
    out = bh_fdr(p, alpha=0.05)
    assert not np.any(out), "no p-value small enough to survive BH"


def test_bonferroni_trap_different_alpha():
    """Different alpha values must produce different rejection masks."""
    rng = np.random.default_rng(20)
    p = rng.uniform(0, 0.15, 30)
    out_strict = bonferroni(p, alpha=0.01)
    out_loose = bonferroni(p, alpha=0.10)
    assert out_strict.sum() <= out_loose.sum(), "stricter alpha must reject ≤ looser alpha"
    # They should differ on this data
    assert not np.array_equal(out_strict, out_loose), "alpha must affect the result"
