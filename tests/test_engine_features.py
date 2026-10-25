"""Engine features the honeypot relies on (ported from the old repo's test_honeypot.py / test_review_fixes.py on 22 Oct,
when the legacy honeypot ladder they lived next to was left behind)."""

import asyncio
import json

import pytest

from collusion.game.engine import Game, fork, make_config, restore, validate_interventions
from collusion.market.economics import PRESETS

TRI = PRESETS["triopoly"]
MARKET = {"a": list(TRI.a), "c": list(TRI.c), "mu": TRI.mu, "a0": TRI.a0, "alpha": TRI.alpha, "beta": TRI.beta}


def _rounds(d):
    return [json.loads(line) for line in open(d / "rounds.jsonl")]


def _cfg(ivs, periods=6):
    plant = {"kind": "scripted", "rule": "plant", "price_rule": "nash", "noise": 0.02, "messages": {"2": "hi"}}
    return make_config(market=MARKET, firms=[{"kind": "llm", "model": "fake:1.6:ok"},
                                             {"kind": "llm", "model": "fake:1.6:ok"}, plant],
                       prompt_mode="chat", periods=periods, seed=3, interventions=ivs)


def test_set_scripted_switch_and_resume(tmp_path):
    ivs = [{"type": "set_scripted", "from_period": 4, "firm": 2,
            "spec": {"price_rule": "fixed", "price": 1.8, "noise": 0.02, "messages": {}}}]
    g = Game(_cfg(ivs, periods=8), tmp_path / "g")
    asyncio.run(g.run())
    r = _rounds(tmp_path / "g")
    nash = TRI.benchmarks()["nash_prices"][2]
    assert all(abs(x["prices"][2] - nash) <= 0.02 + 1e-9 for x in r[1:4])
    assert all(abs(x["prices"][2] - 1.8) <= 0.02 + 1e-9 for x in r[4:])
    assert len({round(x["prices"][2], 6) for x in r[4:]}) > 1  # noisy, not a flat giveaway
    assert r[2]["sent"][2] == "hi"
    # resume from period 5 into a copy: the switch is re-applied and every later row is identical
    g2 = restore(tmp_path / "g", 5, tmp_path / "g2", parent={"src": "g", "at_period": 5})
    asyncio.run(g2.run())
    r2 = _rounds(tmp_path / "g2")
    assert [x["prices"] for x in r2[6:]] == [x["prices"] for x in r[6:]]


def test_set_scripted_validation(tmp_path):
    with pytest.raises(ValueError, match="not a scripted firm"):
        Game(_cfg([{"type": "set_scripted", "from_period": 2, "firm": 0, "spec": {"price": 1.8}}]), tmp_path / "a")
    with pytest.raises(ValueError, match="kind or rule"):
        Game(_cfg([{"type": "set_scripted", "from_period": 2, "firm": 2, "spec": {"rule": "fixed"}}]), tmp_path / "b")
    with pytest.raises(ValueError, match="after the fork point"):
        validate_interventions([{"type": "set_scripted", "from_period": 10, "firm": 2, "spec": {"price": 1.8}}],
                               3, 20, fork_at=10)


def test_overwrite_clears_every_per_game_file(tmp_path):
    """An overwrite must not leave the old game's plant/confederate calls or post-hoc answers behind."""
    stale = ("plant.jsonl", "confederate.jsonl", "posthoc.json")
    asyncio.run(Game(_cfg([]), tmp_path).run())
    for n in stale:
        (tmp_path / n).write_text("{}\n")
    asyncio.run(Game(_cfg([]), tmp_path, overwrite=True).run())
    assert not any((tmp_path / n).exists() for n in stale)
    assert len(_rounds(tmp_path)) == 7  # periods 0..6 of the new game only


def test_fork_parse_failures_exclude_trunk(tmp_path):
    duo = PRESETS["duopoly"]
    market = {"a": list(duo.a), "c": list(duo.c), "mu": duo.mu, "a0": duo.a0, "alpha": duo.alpha, "beta": duo.beta}
    cfg = make_config(market=market, firms=[{"kind": "llm", "model": "fake-raw:no price here"},
                                            {"kind": "scripted", "rule": "nash"}], periods=6, seed=1,
                      prompt_mode="chat")
    asyncio.run(Game(cfg, tmp_path / "trunk").run())
    assert json.loads((tmp_path / "trunk" / "summary.json").read_text())["parse_failures"] == 6
    (d,) = asyncio.run(fork(tmp_path / "trunk", at=4, branches=1, interventions=[], out_root=tmp_path / "f",
                            label="c"))
    assert json.loads((d / "summary.json").read_text())["parse_failures"] == 2


def test_one_failing_game_does_not_kill_others(tmp_path):
    """Self-play gathers its games and reports failures (report_failures); one bad game must not stop the others."""
    from collusion.clients.llm import LLM
    from collusion.game.engine import report_failures

    duo = PRESETS["duopoly"]
    market = {"a": list(duo.a), "c": list(duo.c), "mu": duo.mu, "a0": duo.a0, "alpha": duo.alpha, "beta": duo.beta}

    def cfg(model):
        return make_config(market=market, firms=[{"kind": "llm", "model": model}, {"kind": "scripted", "rule": "nash"}],
                           periods=4, seed=1, prompt_mode="chat")

    good = Game(cfg("fake:1.7"), tmp_path / "good", llm=LLM())
    bad = Game(cfg("nonexistent-model"), tmp_path / "bad", llm=LLM())

    async def both():
        return await asyncio.gather(bad.run(), good.run(), return_exceptions=True)

    failed = report_failures([bad, good], asyncio.run(both()))
    assert [g for g, _ in failed] == [bad]
    assert json.loads((tmp_path / "good" / "summary.json").read_text())["periods"] == 4
    bad_summary = json.loads((tmp_path / "bad" / "summary.json").read_text())
    assert bad_summary["error"] and bad_summary["call_errors"] == 1
    assert "error" in json.loads((tmp_path / "bad" / "calls.jsonl").read_text().splitlines()[0])
