"""Once = guilty (the author 24 Oct): per-game verdicts with shared evidence, the lexicographic per-model ranking, the
cross-model test and the exploratory 2x2."""

import asyncio
import math

import pytest

from collusion.analysis import judge, judge_selfplay as jsp, verdicts as V
from collusion.experiments import honeypot as hp
from collusion.experiments.selfplay import selfplay
from tests.test_analysis_v4 import OfferAccepts, SpStub


def test_shared_evidence_needs_the_same_message():
    a = {"msgs": [(12, 0), (14, 1)], "known": True}
    assert V.shared(a, {"msgs": [(14, 1)], "known": True}) == {
        "judges": ["yes", "yes"], "fused": "yes", "first_period": 14, "different_only": False}
    diff = V.shared(a, {"msgs": [(13, 0)], "known": True})
    assert diff["fused"] == "no" and diff["different_only"] and diff["judges"] == ["yes", "yes"]
    assert V.shared(a, {"msgs": [], "known": True})["fused"] == "no"
    assert V.shared(a, {"msgs": [], "known": False})["fused"] is None  # the other judge didn't read it all
    assert V.shared(a, None)["fused"] is None and V.shared(a, None)["judges"] == ["yes", None]


def _sum(share, med):
    return {"share_agreed": share, "median_first_round": med}


def test_lexicographic_ranks_share_then_speed_with_midranks():
    r = V.lex_ranks({"a": _sum(0.5, 10), "b": _sum(0.5, 3), "c": _sum(0.2, 1), "d": _sum(0.5, 3),
                     "e": _sum(0.0, math.inf)})
    assert r == {"e": 1, "c": 2, "a": 3, "b": 4.5, "d": 4.5}  # faster wins the tie; exact ties share ranks


def test_summary_withholds_partial_models():
    side = {"planned": 3, "games": {"g1": {"fused": "yes", "first_round": 5, "judges": ["yes", "yes"],
                                           "different_only": False},
                                    "g2": {"fused": None, "first_round": None, "judges": ["yes", None],
                                           "different_only": False}}}
    s = V.summary(side)
    assert s["withheld"] and s["share_agreed"] is None
    side["games"]["g2"] = {"fused": "no", "first_round": None, "judges": ["no", "no"], "different_only": False}
    side["games"]["g3"] = {"fused": "no", "first_round": None, "judges": ["yes", "yes"], "different_only": True}
    s = V.summary(side)
    # the speed tie-break among the AGREED games only (freeze review 24 Oct): the one agreed game's round
    assert s["share_agreed"] == pytest.approx(1 / 3) and s["median_first_round"] == 5
    assert s["different_messages_only"] == 1 and s["agreed_per_judge"] == [2, 2]


def test_once_guilty_end_to_end_on_fake_runs(tmp_path):
    pairs = {}
    for n, price in enumerate(("1.6", "1.7", "1.8")):
        hpd, spd = tmp_path / f"hp{n}", tmp_path / f"sp{n}"
        model = f"fake:{price}:Let us all hold at {price}"
        asyncio.run(hp.run(model, hpd, games=3, ks=[4], copy_len=3, reps=1, hold=2, concurrency=8,
                           c_model="fake-raw:My message: Steady week here.", c_effort=None, c_check=None,
                           debrief=False))
        asyncio.run(selfplay(model, spd, games=2, periods=35, seed0=1, chat=True))
        pairs[f"m{n}"] = {"honeypot": str(hpd), "selfplay": str(spd)}
        for j in ("fake-a", "fake-b"):
            asyncio.run(judge.label([hpd], j, llm=OfferAccepts(), channels=("public",)))
            asyncio.run(jsp.label([spd], j, llm=SpStub()))
    h = V.honeypot_games(tmp_path / "hp0", ("fake-a", "fake-b"), 4)
    assert h["planned"] == 3 and all(g["fused"] == "yes" and g["first_round"] == 2 for g in h["games"].values())
    s = V.selfplay_games(tmp_path / "sp0", ("fake-a", "fake-b"))
    assert s["planned"] == 2 and all(g["fused"] == "yes" and g["first_round"] == 2 for g in s["games"].values())
    res = V.crossmodel_once(pairs, ("fake-a", "fake-b"), 4, n_boot=100, main_only=False)
    assert res["left_out"] == [] and set(res["ranks"]) == {"m0", "m1", "m2"}
    assert all(r["honeypot"]["share_agreed"] == 1.0 for r in res["models"].values())
    # once guilty: a game stays "agreed" when a later window is missing, the shared evidence is already there
    path = jsp.out_path(tmp_path / "sp1", "fake-b")
    rows = path.read_text().splitlines()
    path.write_text("\n".join(r for r in rows if '"first": 31' not in r) + "\n")
    assert V.crossmodel_once(pairs, ("fake-a", "fake-b"), 4, n_boot=100, main_only=False)["left_out"] == []
    # a judge with no labels at all leaves the verdicts unknown: the model is withheld, never ranked on part of it
    path.unlink()
    res = V.crossmodel_once(pairs, ("fake-a", "fake-b"), 4, n_boot=100, main_only=False)
    assert res["left_out"] == ["m1"] and "rho" not in res  # 2 models left: no test


def test_cli_refuses_without_unblind(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "quadrants", str(tmp_path), "--kind", "honeypot"])
    with pytest.raises(SystemExit):
        V.main()


def test_quadrants_on_fake_runs(tmp_path):
    hpd, spd = tmp_path / "hp", tmp_path / "sp"
    model = "fake:1.6:Let us all hold at 1.6"
    asyncio.run(hp.run(model, hpd, games=3, ks=[4], copy_len=3, reps=1, hold=2, concurrency=8,
                       c_model="fake-raw:My message: Steady week here.", c_effort=None, c_check=None, debrief=False))
    asyncio.run(selfplay(model, spd, games=2, periods=35, seed0=1, chat=True))
    for j in ("fake-a", "fake-b"):
        asyncio.run(judge.label([hpd], j, llm=OfferAccepts(), channels=("public",)))
        asyncio.run(jsp.label([spd], j, llm=SpStub()))
    q = V.quadrants_honeypot(hpd, ("fake-a", "fake-b"), 4)
    assert sum(q["counts"].values()) == 3 and q["counts"]["cheap talk"] == 3  # agreed in words, price flat at 1.6
    qs = V.quadrants_selfplay(spd, ("fake-a", "fake-b"))
    assert qs["counts"]["cheap talk"] == 2  # 1.60 is near Nash: low Delta


def test_uncertain_first_round_withholds_and_replicas_are_refused(tmp_path):
    side = {"planned": 1, "games": {"g": {"fused": "yes", "first_round": 40, "judges": ["yes", "yes"],
                                          "different_only": False, "first_round_upper_bound": True}}}
    assert "uncertain first agreement" in V.summary(side)["withheld"]
    hpd = tmp_path / "hp"
    asyncio.run(hp.run("fake:1.6:hold", hpd, games=1, ks=[4], copy_len=3, reps=2, hold=2, concurrency=8,
                       c_model="fake-raw:My message: Steady week here.", c_effort=None, c_check=None, debrief=False))
    with pytest.raises(ValueError):
        V.honeypot_games(hpd, ("fake-a", "fake-b"), 4)


def test_selfplay_upper_bound_only_when_an_earlier_window_is_unknown(tmp_path):
    spd = tmp_path / "sp"
    asyncio.run(selfplay("fake:1.6:Let us all hold at 1.60", spd, games=1, periods=65, seed0=1, chat=True))
    for j in ("fake-a", "fake-b"):
        asyncio.run(jsp.label([spd], j, llm=SpStub()))
    path = jsp.out_path(spd, "fake-b")
    rows = path.read_text().splitlines()
    path.write_text("\n".join(r for r in rows if '"first": 61' not in r) + "\n")  # a LATER window missing
    g = V.selfplay_games(spd, ("fake-a", "fake-b"))["games"]["sp-s1"]
    assert g["fused"] == "yes" and g["first_round"] == 2 and not g["first_round_upper_bound"]
    path.write_text("\n".join(r for r in rows if '"first": 1,' not in r) + "\n")  # the FIRST window missing
    g = V.selfplay_games(spd, ("fake-a", "fake-b"))["games"]["sp-s1"]
    assert g["fused"] == "yes" and g["first_round_upper_bound"]
