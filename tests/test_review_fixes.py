"""Regression tests for the issues found in the adversarial code and fidelity reviews."""

import asyncio
import json

import pytest

from collusion.clients.costs import call_cost
from collusion.market.economics import PRESETS
from collusion.game.engine import Game, fork, load_rounds, make_config, restore
from collusion.game.prompts import build_chat, full_text, parse_message, parse_price, parse_price_lcartel


def market_dict(name="duopoly"):
    m = PRESETS[name]
    return {"a": list(m.a), "c": list(m.c), "mu": m.mu, "a0": m.a0, "alpha": m.alpha, "beta": m.beta}


def scripted_cfg(**kw):
    firms = [{"kind": "scripted", "rule": "nash", "noise": 0.05}, {"kind": "scripted", "rule": "tit_for_tat"}]
    return make_config(market=market_dict(), firms=firms, periods=20, seed=7, **kw)


def llm_cfg(model, mode="chat", periods=3, **kw):
    firms = [{"kind": "llm", "model": model}, {"kind": "scripted", "rule": "nash"}]
    return make_config(market=market_dict(), firms=firms, periods=periods, seed=1, prompt_mode=mode, **kw)


def calls_of(d):
    return [json.loads(line) for line in open(d / "calls.jsonl")]


# ---------- 1. truncated replies ----------
def test_strict_parser_ignores_unclosed_reasoning_and_odd_numbers():
    assert parse_price("<think>\n... so My chosen price: 1.9 would be") is None
    assert parse_price("<think>a</think>My chosen price: 1.756") == 1.76
    assert parse_price("My chosen price: 1,75") == 1.75
    assert parse_price("My chosen price: 1,750") is None
    assert parse_price("My chosen price: 175 cents") is None


def test_strict_mode_retries_and_flags_cut_off_replies(tmp_path):
    g = asyncio.run(Game(llm_cfg("fake-cut:My chosen price: 1.9"), tmp_path / "g").run())
    r1 = load_rounds(tmp_path / "g")[1]
    assert r1["parse_failed"][0] and r1["truncated"][0]
    assert r1["prices"][0] == load_rounds(tmp_path / "g")[0]["prices"][0]  # documented fallback
    attempts = [c for c in calls_of(tmp_path / "g") if c["period"] == 1]
    assert len(attempts) == 3 and all(c["truncated"] and c["price"] is None for c in attempts)
    assert json.loads((tmp_path / "g" / "summary.json").read_text())["truncated_decisions"] == 3
    assert g.cfg["parse_mode"] == "strict"


def test_lcartel_parser_matches_their_rule():
    # first attempt: first parseable line, case-sensitive, no $ or ** allowed
    assert parse_price_lcartel("x</think>\nMy chosen price: 1.756\nMy chosen price: 1.9", 1) == 1.76
    assert parse_price_lcartel("x</think>My chosen price: $1.80", 1) is None
    assert parse_price_lcartel("x</think>My chosen price: **1.8**", 1) is None
    assert parse_price_lcartel("x</think>My Chosen Price: 1.8", 1) is None
    assert parse_price_lcartel("x</think>my chosen price: 1.8", 1) == 1.8  # substring "chosen price:" matches
    # retries (their getAction): last line wins, any unparsable "chosen price:" line fails the attempt
    assert parse_price_lcartel("x</think>My chosen price: 1.7\nMy chosen price: 1.8", 2) == 1.8
    assert parse_price_lcartel("x</think>My chosen price: 1.7\nMy chosen price: about 1.8", 2) is None
    # cut off before </think>: the whole text, reasoning included, is searched (as in their runs)
    assert parse_price_lcartel("thinking...\nMy chosen price: 1.9", 1) == 1.9
    assert parse_price_lcartel("thinking... so My chosen price: 1.9 would be", 1) is None


def test_full_text_rebuilds_reasoning_parser_split():
    assert full_text("My chosen price: 1.8", "think") == "think</think>My chosen price: 1.8"
    assert full_text("", "cut off mid thought") == "cut off mid thought"
    assert full_text("plain", None) == "plain"


def test_lcartel_mode_keeps_cut_off_price_but_flags_it(tmp_path):
    asyncio.run(Game(llm_cfg("fake-cut:reasoning\\nMy chosen price: 1.9", mode="p0"), tmp_path / "g").run())
    r1 = load_rounds(tmp_path / "g")[1]
    assert r1["prices"][0] == 1.9 and r1["truncated"][0] and not r1["parse_failed"][0]


# ---------- context cap / sampling (fidelity) ----------
def test_p0_defaults_reproduce_llmcartel_sampling(tmp_path):
    cfg = llm_cfg("fake:1.7", mode="p0")
    assert (cfg["top_p"], cfg["context_cap"], cfg["parse_mode"], cfg["retries"]) == (1.0, 8192, "lcartel", 9)
    chat = llm_cfg("fake:1.7")
    assert (chat["top_p"], chat["context_cap"], chat["parse_mode"], chat["retries"]) == (None, None, "strict", 3)


def test_context_cap_clamps_max_tokens(tmp_path):
    asyncio.run(Game(llm_cfg("fake:1.7", mode="p0", context_cap=300), tmp_path / "g").run())
    for c in calls_of(tmp_path / "g"):
        assert c["max_tokens"] == max(1, min(4096, 300 - c["prompt_tokens_counted"]))
        assert c["usage"]["requested_max_tokens"] == c["max_tokens"]
    asyncio.run(Game(llm_cfg("fake:1.7", mode="p0", context_cap=0), tmp_path / "off").run())
    assert all(c["max_tokens"] == 4096 for c in calls_of(tmp_path / "off"))


# ---------- 2. existing directories ----------
def test_rerun_into_existing_dir_refused(tmp_path):
    asyncio.run(Game(scripted_cfg(), tmp_path / "g").run())
    with pytest.raises(FileExistsError):
        asyncio.run(Game(scripted_cfg(), tmp_path / "g").run())
    asyncio.run(Game(scripted_cfg(), tmp_path / "g", overwrite=True).run())
    assert [r["period"] for r in load_rounds(tmp_path / "g")] == list(range(21))


def test_fork_rerun_refused(tmp_path):
    asyncio.run(Game(scripted_cfg(), tmp_path / "trunk").run())
    asyncio.run(fork(tmp_path / "trunk", at=5, branches=1, interventions=[], out_root=tmp_path / "f", label="c"))
    with pytest.raises(FileExistsError):
        asyncio.run(fork(tmp_path / "trunk", at=5, branches=1, interventions=[], out_root=tmp_path / "f", label="c"))


# ---------- 3. intervention validation ----------
@pytest.mark.parametrize("iv", [
    {"type": "force_prices", "period": 6, "firm": 0, "price": "nash"},
    {"type": "force_price", "period": 6, "firm": "0", "price": "nash"},
    {"type": "force_price", "period": 6, "firm": 2, "price": "nash"},
    {"type": "force_price", "period": 5, "firm": 0, "price": "nash"},  # not after the fork point
    {"type": "chat_delivery", "from_period": 4, "on": False},
    {"type": "chat_delivery", "from_period": 8, "on": "no"},
    {"type": "inject_message", "period": 3, "from": 1, "text": "hi"},  # before the fork point
    {"type": "inject_message", "period": 6, "from": 1, "to": ["0"], "text": "hi"},
])
def test_bad_interventions_fail_loudly(tmp_path, iv):
    asyncio.run(Game(scripted_cfg(), tmp_path / "trunk").run())
    with pytest.raises(ValueError):
        asyncio.run(fork(tmp_path / "trunk", at=5, branches=1, interventions=[iv], out_root=tmp_path / "f",
                         label="x"))


def test_bad_intervention_in_fresh_game_fails(tmp_path):
    with pytest.raises(ValueError):
        Game(scripted_cfg(interventions=[{"type": "force_price", "period": 99, "firm": 0, "price": 1.5}]),
             tmp_path / "g")


# ---------- 4. fork beyond the trunk ----------
def test_fork_past_trunk_end_errors(tmp_path):
    asyncio.run(Game(scripted_cfg(), tmp_path / "trunk").run(until=4))
    with pytest.raises(ValueError):
        asyncio.run(fork(tmp_path / "trunk", at=8, branches=1, interventions=[], out_root=tmp_path / "f", label="x"))
    with pytest.raises(ValueError):
        restore(tmp_path / "trunk", 8, tmp_path / "copy")


# ---------- should-fix ----------
def test_message_is_one_line_and_none_variants():
    assert parse_message("My message: Let's hold 1.80.\nI say this because the rival...") == "Let's hold 1.80."
    assert parse_message("My message: NONE (no message this round)") is None
    assert parse_message("My message: None.") is None
    assert parse_message("<think>My message: secret</think>My chosen price: 1.7") is None


def test_snapshot_model_ids_are_priced():
    u = {"prompt_tokens": 1000, "completion_tokens": 100}
    assert call_cost("gpt-4.1-2025-04-14", u) == call_cost("gpt-4.1", u) is not None
    assert call_cost("qwen3.8-27b-fp8", u) is None


def test_triopoly_chat_labels_prices_like_messages():
    hist = [{"period": 0, "prices": [1.5, 1.6, 1.8], "quantities": [30, 30, 30], "profits": [15, 18, 24],
             "sent": [None, None, "hold 1.80"], "inbox": [{"2": "hold 1.80"}, {"2": "hold 1.80"}, {}]}]
    user = build_chat(hist, me=0, cost=1.0, ceiling=2.89, window=100)[1]["content"]
    assert "   - Firm B's price: 1.6" in user and "   - Firm C's price: 1.8" in user
    assert "   - Message from Firm C: hold 1.80" in user


def test_triopoly_names_mean_the_same_firm_for_every_reader():
    """Prompt v2: a broadcast message naming "Firm B" must mean firm 1 to every reader (v1 used reader-relative
    "Competitor A/B", so the same words named different firms)."""
    hist = [{"period": 0, "prices": [1.5, 1.6, 1.8], "quantities": [30, 30, 30], "profits": [15, 18, 24],
             "sent": ["Firm B, hold 1.60", None, None], "inbox": [{}, {"0": "Firm B, hold 1.60"}, {"0": "Firm B, hold 1.60"}]}]
    for me in range(3):
        user = build_chat(hist, me=me, cost=1.0, ceiling=2.89, window=100)[1]["content"]
        assert f"my firm is Firm {'ABC'[me]};" in user
        assert "Competitor A" not in user and "Competitor B" not in user
        for j in range(3):
            if j != me:
                assert f"   - Firm {'ABC'[j]}'s price: {hist[0]['prices'][j]}" in user
        if me != 0:
            assert "   - Message from Firm A: Firm B, hold 1.60" in user
        assert "under the round in which they were sent" in user


def test_injected_message_bypasses_delivery_off(tmp_path):
    cfg = llm_cfg("fake:1.7", periods=4, chat_delivery=False,
                  interventions=[{"type": "inject_message", "period": 2, "from": 1, "to": [0], "text": "hold 1.9"}])
    asyncio.run(Game(cfg, tmp_path / "g").run())
    assert load_rounds(tmp_path / "g")[2]["inbox"][0] == {"1": "hold 1.9"}
