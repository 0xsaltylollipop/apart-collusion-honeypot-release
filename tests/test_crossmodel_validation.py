"""Known-answer checks of crossmodel's rho and exact one-sided p against scipy.stats (spearmanr; permutation_test with
exact enumeration of all n! pairings), on hand-made datasets with and without ties, and the n = 4 examples worked by
hand in docs/reviews/crossmodel-validation-2026-10-24.md. Synthetic data only."""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import permutation_test, rankdata, spearmanr

from collusion.analysis import crossmodel as cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _rank_corr(x, y, axis):
    rx, ry = rankdata(x, axis=axis), rankdata(y, axis=axis)
    rx, ry = rx - rx.mean(axis=axis, keepdims=True), ry - ry.mean(axis=axis, keepdims=True)
    return (rx * ry).sum(axis=axis) / np.sqrt((rx ** 2).sum(axis=axis) * (ry ** 2).sum(axis=axis))


def _scipy_p(x, y):
    """scipy's exact one-sided permutation p: every ordering of x against a fixed y (n! of them), share with rho >=
    observed. One sample only, as scipy's docs do for a correlation: with both, "pairings" permutes each, (n!)^2."""
    y = np.asarray(y, dtype=float)
    return permutation_test((np.asarray(x, dtype=float),), lambda a, axis: _rank_corr(a, np.broadcast_to(y, a.shape),
                                                                                       axis),
                            permutation_type="pairings", alternative="greater", n_resamples=math.inf,
                            vectorized=True).pvalue


DATASETS = {
    "n4 no ties": ([1, 2, 3, 4], [1, 3, 2, 4]),
    "n4 tie in y": ([1, 2, 3, 4], [10, 20, 20, 40]),
    "n5 ties both": ([0.0, 0.0, 0.2, 0.5, 0.5], [0.3, 0.1, 0.3, 0.9, 0.3]),
    "n6 negative": ([6, 5, 4, 3, 2, 1], [1, 2, 3, 5, 4, 6]),
    "n7 at-cap pile-up": ([0.0, 0.0, 0.0, 0.1, 0.3, 0.4, 0.6], [0.2, 0.31, 0.25, 0.99, 0.99, 0.99, 0.5]),
    "n8 mixed": ([0.12, 0.5, 0.33, 0.33, 0.9, 0.05, 0.7, 0.2], [0.4, 0.6, 0.1, 0.35, 0.8, 0.2, 0.8, 0.3]),
    "n9 three at cap, three at 0": ([0, 0, 0, 0.2, 0.25, 0.4, 0.5, 0.55, 0.8],
                                    [0.1, 0.4, 0.2, 0.3, 0.99, 0.5, 0.99, 0.99, 0.6]),
    "n9 no ties": ([0.3, 0.1, 0.8, 0.5, 0.45, 0.9, 0.2, 0.65, 0.05], [0.2, 0.3, 0.7, 0.4, 0.6, 0.95, 0.1, 0.5, 0.15]),
}


@pytest.mark.parametrize("name", list(DATASETS))
def test_rho_and_exact_p_match_scipy(name):
    x, y = DATASETS[name]
    assert cm.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)
    r = cm.perm_p(x, y)
    assert r["method"] == "exact" and r["orderings"] == math.factorial(len(x))
    assert r["p"] == pytest.approx(_scipy_p(x, y), abs=1e-12)


def test_random_tied_datasets_match_scipy():
    rng = random.Random(11)
    for _ in range(25):
        n = rng.randint(4, 7)
        x, y = [rng.randint(0, 3) / 4 for _ in range(n)], [rng.randint(0, 4) / 4 for _ in range(n)]
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        assert cm.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)
        assert cm.perm_p(x, y)["p"] == pytest.approx(_scipy_p(x, y), abs=1e-12)


def test_hand_computed_n4_examples():
    # docs/reviews/crossmodel-validation-2026-10-24.md, "A tiny example by hand"
    # no ties: sum d^2 = 2, rho = 1 - 6 * 2 / (4 * 15) = 0.8; orderings with sum d^2 <= 2: identity + 3 adjacent swaps
    assert cm.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert cm.perm_p([1, 2, 3, 4], [1, 3, 2, 4])["p"] == 4 / 24
    # a tie: y ranks (1, 2.5, 2.5, 4); rho = 4.5 / sqrt(5 * 4.5) = 0.9487; only the observed sum 29.5 (reached by 2
    # of the 24 orderings: the tied pair swapped) is the maximum, so p = 2 / 24
    assert cm.spearman([1, 2, 3, 4], [10, 20, 20, 40]) == pytest.approx(4.5 / math.sqrt(22.5))
    assert cm.perm_p([1, 2, 3, 4], [10, 20, 20, 40])["p"] == 2 / 24


def test_synthetic_records_have_collects_schema():
    # the simulation study's records go through crossmodel() exactly like collect()'s
    import validate_crossmodel as v
    v.check_schema()
    recs, _ = v.draw(random.Random(1), 0.5, **v.SCENARIOS["base"])
    res = cm.crossmodel(recs, h1="offer_gap", n_boot=50)
    # the study tests offer_gap on purpose: the only problem is the loud predictor override (final re-review 24 Oct)
    assert [q for q in res["problems"] if not q.startswith("H2 PREDICTOR OVERRIDDEN")] == []
    assert res["H2"]["n"] == 9 and not res["H2"]["provisional"]
