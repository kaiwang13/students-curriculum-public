import numpy as np
import statsmodels.api as sm
from solution import assoc


def test_matches_statsmodels():
    rng = np.random.default_rng(0)
    n = 200
    cov = rng.normal(size=(n, 2))
    G = rng.integers(0, 3, size=(n, 3)).astype(float)
    y = 0.5 * G[:, 0] + 0.3 * cov[:, 0] + rng.normal(size=n)
    out = assoc(G, y, covariates=cov)
    # full-model OLS for SNP 0 as oracle
    X = sm.add_constant(np.column_stack([cov, G[:, 0]]))
    res = sm.OLS(y, X).fit()
    assert np.isclose(out["beta"][0], res.params[-1])
    assert np.isclose(out["se"][0], res.bse[-1])
    assert np.isclose(out["p"][0], res.pvalues[-1])
