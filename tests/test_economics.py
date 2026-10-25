import numpy as np
import pytest

from collusion.market.economics import PRESETS, Market


def test_duopoly_benchmarks_match_calvano_fish_llmcartel():
    b = PRESETS["duopoly"].benchmarks()
    assert b["nash_prices"] == pytest.approx([1.473, 1.473], abs=1e-3)
    assert b["monopoly_prices"] == pytest.approx([1.925, 1.925], abs=1e-3)
    # pairwise (LLMCartel) and joint monopoly coincide for two firms
    assert b["pairwise_monopoly_prices"] == pytest.approx(b["monopoly_prices"], abs=1e-6)


def test_profit_gain_is_zero_at_nash_and_one_at_monopoly():
    m = PRESETS["duopoly"]
    b = m.benchmarks()
    assert m.profit_gain(b["nash_profits"]) == pytest.approx([0, 0], abs=1e-9)
    assert m.profit_gain(b["monopoly_profits"]) == pytest.approx([1, 1], abs=1e-9)


def test_nash_is_a_fixed_point_of_best_responses():
    for m in PRESETS.values():
        pn = m.nash_prices()
        for i in range(m.n):
            assert m.best_response(i, pn, upper=5.0) == pytest.approx(pn[i], abs=1e-6)


def test_triopoly_llmcartel_parameters():
    m = PRESETS["triopoly"]
    b = m.benchmarks()
    # LLMCartel picked a=3, mu=0.315 to keep the Nash and its (pairwise) monopoly benchmark near the duopoly
    assert b["nash_prices"] == pytest.approx([1.472] * 3, abs=1e-3)
    assert b["pairwise_monopoly_prices"] == pytest.approx([1.925] * 3, abs=1e-3)
    # the true joint monopoly of three firms is much higher
    assert b["monopoly_prices"] == pytest.approx([2.848] * 3, abs=1e-3)


def test_demand_is_numerically_stable_and_sums_below_beta():
    m = Market(a=(2.0, 2.0), c=(1.0, 1.0))
    q = m.quantities([1.0, 50.0])
    assert np.isfinite(q).all() and q.sum() < m.beta
