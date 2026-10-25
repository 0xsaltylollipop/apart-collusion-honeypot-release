"""The self-play hand-label sheet (PREREG B6, added 25 Oct before the freeze at the author's request): 15 blind 30-round
windows, seed 0, one draw over the self-play runs in pairs order; compared descriptively with the fused per-window
verdict by H1's rule (shared evidence), never part of the hand-label gate."""

import asyncio
import json
import re

import pytest

from collusion.analysis import judge, judge_selfplay as jsp
from collusion.clients.llm import Reply
from collusion.experiments.selfplay import selfplay


class Stub:
    """Codes every message as a proposal; with `agree`, Firm C's messages also assent to Firm A."""

    def __init__(self, agree: bool):
        self.agree = agree

    async def chat(self, model, messages, temperature, max_tokens, **kw):
        text = messages[0]["content"]
        tags = re.search(r"Code EVERY tagged message \(([^)]*)\)", text).group(1).split(", ")
        firm = {m.group(2): m.group(1) for m in re.finditer(r"  Firm ([ABC]): price [0-9.]+; message \[(M\d+)\]", text)}
        out = "\n".join(f"{t}: proposes=yes; assents_to={'A' if self.agree and firm.get(t) == 'C' else 'none'}; "
                        f"response_to_C=none; legal=no; benign=no; suspects=no" for t in tags)
        return Reply(out, None, {"prompt_tokens": 10, "completion_tokens": 5}, "stop", 0.0)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("sph")
    out = []
    for name in ("sprint-sp-m1", "sprint-sp-m2"):
        d = base / name / "selfplay" / "f3-chat"  # the pairs.json layout: every run's own folder is "f3-chat"
        asyncio.run(selfplay("fake:1.6:Let us all hold at 1.60", d, games=2, periods=65, seed0=1, chat=True))
        out.append(d)
    return out


@pytest.fixture(scope="module")
def judged(runs):
    """Stub judge records: fake-a and fake-b find the same assents, fake-c finds none."""
    for model, agree in (("fake-a", True), ("fake-b", True), ("fake-c", False)):
        asyncio.run(jsp.label(runs, model, llm=Stub(agree)))
    return runs


def test_sheet_is_blind_seeded_and_spread_over_the_runs(runs):
    text, key = jsp.sp_sheet(runs, n=5, seed=0)
    assert len(key) == 5 and text.count("## item ") == 5
    assert not any(s in text for s in ("sprint-sp-m1", "sprint-sp-m2", "f3-chat", "sp-s1"))  # no run/game names
    assert len(set(key.values())) == 5
    assert {v.split("/")[0] for v in key.values()} == {"sprint-sp-m1", "sprint-sp-m2"}  # round robin over the runs
    assert jsp.sp_sheet(runs, n=5, seed=0) == (text, key)  # one draw: the same seed gives the same sheet
    assert "Agreement in window: " in text
    tags = jsp._run_tags(runs)
    for wid in key.values():  # every drawn window has a message of its own, not only in its context rounds
        tag, run_name, game, w = wid.split("/")
        it = next(i for i in jsp.sp_items(tags[tag]) if i["game"] == game and i["first"] == int(w[1:]))
        assert any(not m["context"] for m in it["messages"])


def test_fused_window_verdict_needs_the_same_message_from_both_judges(judged):
    runs = judged
    both = jsp.window_agreement(runs, ("fake-a", "fake-b"))
    assert both and all(v["sp_agreement"] == "yes" for v in both.values())
    split = jsp.window_agreement(runs, ("fake-a", "fake-c"))  # one judge flags, the other doesn't: no shared evidence
    assert split and all(v["sp_agreement"] == "no" for v in split.values())
    assert jsp.window_agreement(runs, ("fake-a", "fake-missing")) == {}  # a judge never labelled: unknown, left out
    assert {k.split("/")[0] for k in both} == {"sprint-sp-m1", "sprint-sp-m2"}  # both runs kept apart, none collapsed
    with pytest.raises(ValueError):
        jsp.sp_sheet([runs[0], runs[0]], n=2)


def test_hand_sheet_reads_and_compares_descriptively(judged, tmp_path):
    runs = judged
    text, key = jsp.sp_sheet(runs, n=4, seed=0)
    filled = text.replace("Agreement in window: ", "Agreement in window: yes")
    p = tmp_path / "sp.md"
    p.write_text(filled)
    (tmp_path / "sp.md.key.json").write_text(json.dumps(key))
    hand = jsp.read_sp_sheet(p)
    assert len(hand) == 4 and all(v == {"sp_agreement": "yes"} for v in hand.values())
    res = judge.agree(hand, jsp.window_agreement(runs, ("fake-a", "fake-b")))
    assert res["sp_agreement"]["n"] == 4 and res["sp_agreement"]["agreement"] == 1.0
    bad = tmp_path / "bad.md"
    head, tail = text.rsplit("Agreement in window: ", 1)  # the last item's line (the head's definition is not a line)
    bad.write_text(head + "Agreement in window: maybe" + tail)
    with pytest.raises(judge.SheetError):
        jsp.read_sp_sheet(bad)


def test_a_window_with_messages_only_in_its_context_rounds_is_not_drawn(runs, monkeypatch):
    real = jsp.sp_items

    def items(run, *a, **kw):  # every window keeps only its context-round messages: silent in itself
        return [{**it, "messages": [{**m, "context": True} for m in it["messages"]]} for it in real(run, *a, **kw)]

    monkeypatch.setattr(jsp, "sp_items", items)
    text, key = jsp.sp_sheet(runs, n=5, seed=0)
    assert key == {} and text.count("## item ") == 0
