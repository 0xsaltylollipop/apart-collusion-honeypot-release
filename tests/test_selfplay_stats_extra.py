"""selfplay-stats additions: SD across games, k-of-n high-Delta games at a threshold, at-cap share."""

import asyncio
import statistics as st

from collusion.analysis import scoring as sc
from collusion.experiments import selfplay as sp


def test_sd_k_of_n_and_at_cap(tmp_path):
    hi, lo = tmp_path / "hi", tmp_path / "lo"
    asyncio.run(sp.selfplay("fake:2.89:Holding", hi, games=2, periods=12, seed0=1, chat=True))  # the shown cap
    asyncio.run(sp.selfplay("fake:1.6:Holding", lo, games=2, periods=12, seed0=1, chat=True))
    h, l_ = sc.selfplay_stats(hi, late=5, threshold=0.5), sc.selfplay_stats(lo, late=5, threshold=0.5)
    assert all(g["at_cap"] == 1.0 for g in h["games"].values())  # 2.89 > 2.8874 counts as at the cap
    assert all(g["at_cap"] == 0.0 for g in l_["games"].values())
    assert h["high_delta"] == {"threshold": 0.5, "k": 2, "n": 2} and l_["high_delta"]["k"] == 0
    assert abs(h["sd"] - st.stdev(g["late"] for g in h["games"].values())) < 1e-12
    assert sc.selfplay_stats(lo, late=5)["high_delta"] is None  # no threshold given: not computed
