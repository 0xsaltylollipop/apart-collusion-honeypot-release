"""The exploratory third judge run through Claude Code subagents (scripts/opus_cc.py): the dumped prompts are the
registered judges' byte for byte, merged replies become records the analysis reads under the third judge's name,
failures stay unknown and are offered again, other judges' files are never touched, and a single judge or the
Luna + third-judge pair runs through H1 as a loud override."""

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from collusion.analysis import judge as J, judge_selfplay as jsp, verdicts as V
from collusion.experiments import honeypot as hp
from collusion.experiments.selfplay import selfplay
from tests.test_analysis_v4 import OfferAccepts, SpStub

_spec = importlib.util.spec_from_file_location("opus_cc", Path(__file__).resolve().parents[1] / "scripts" / "opus_cc.py")
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)
CC, LUNA, HAIKU = O.JUDGE_CC, *J.JUDGES
K = 4


def _stub_reply(text: str) -> str:
    stub = SpStub() if "All three firms are under review" in text else OfferAccepts()
    return asyncio.run(stub.chat("x", [{"role": "user", "content": text}], None, 100)).content


def _transcript(model: str | None, pf: Path, variant: str | None = None) -> list[dict]:
    """A subagent transcript shaped like the real ones: the API request and every turn carry the effort; one Read of
    the prompt file whose result has the Read tool's line-number prefixes; StructuredOutput; and a <synthetic>
    harness turn, which never counts. variant: effort (high instead of low) | low (low instead of high) | high (all
    high) | bash | offset | otherfile | large (allow_large, accepted)."""
    eff = "high" if variant in ("effort", "high") else "low"
    text = pf.read_text(encoding="utf-8") if pf.exists() else ""
    numbered = "\n".join(f"{i:6d}\t{x}" for i, x in enumerate(text.split("\n"), 1))
    read = {"file_path": str(pf)}
    if variant == "offset":
        read["offset"] = 3
    if variant == "large":
        read["allow_large"] = True
    if variant == "otherfile":
        read = {"file_path": str(pf.parent.parent / O.KEY_DIR / "manifest.json")}
    turn = lambda content: {"type": "assistant", "perTurnEffort": eff, "message": {"model": model, "content": content}}
    rows = [{"type": "api-request", "params": {"model": model, "output_config": {"effort": eff}}},
            turn([{"type": "tool_use", "id": "t1", "name": "Read", "input": read}]),
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": numbered}]}}]
    if variant == "bash":
        rows.append(turn([{"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls"}}]))
    rows.append(turn([{"type": "tool_use", "id": "t3", "name": "StructuredOutput", "input": {"reply": "x"}}]))
    rows.append({"type": "assistant", "message": {"model": "<synthetic>", "content": []}})
    return rows


def _journal(d: Path, entries: list[tuple]) -> Path:
    """A Workflow journal like the real one: started + result per agent (agent ids unique per journal folder), and
    each agent's transcript (prompts in d/../prompts). An entry is (label, reply, model[, result_agent, result_model
    [, variant]]): the result can come from another agentId with its own transcript model (None: no transcript)."""
    d.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "launched"}]
    for n, (label, reply, model, *other) in enumerate(entries):
        key, aid = f"v2:{d.name}:{n:04d}", f"a{d.name}x{n:04d}"
        rid, rmodel, variant = (other + [None, None, None])[:3]
        pf = d.parent / "prompts" / f"{label}.txt"
        lines.append({"type": "started", "key": key, "agentId": aid, "label": label, "phase": "Label"})
        (d / f"agent-{aid}.jsonl").write_text("\n".join(json.dumps(x) for x in _transcript(model, pf, variant)) + "\n")
        if rid:
            if rmodel is not None:
                (d / f"agent-{rid}.jsonl").write_text("\n".join(json.dumps(x) for x in _transcript(rmodel, pf)) + "\n")
        else:
            rid = aid
        if reply is not None:
            lines.append({"type": "result", "key": key, "agentId": rid, "result": {"reply": reply}})
    (d / "journal.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return d / "journal.jsonl"


def _hashes(root: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()
            and "claude-opus-5-5-cc" not in p.name and p.name != ".opus_cc.lock"}  # (merge's empty lock files)


def _world(root: Path, models: tuple[str, ...]) -> dict:
    pairs = {}
    for n, model in enumerate(models):
        hpd, spd = root / f"hp{n}", root / f"sp{n}"
        asyncio.run(hp.run(model, hpd, games=3, ks=[K], copy_len=3, reps=1, hold=2, concurrency=8,
                           c_model="fake-raw:My message: Steady week here.", c_effort=None, c_check=None,
                           debrief=False))
        asyncio.run(selfplay(model, spd, games=2, periods=35, seed0=1, chat=True))
        pairs[f"m{n}"] = {"honeypot": str(hpd), "selfplay": str(spd)}
        for j in J.JUDGES:  # the registered judges' names, stub replies (their prompt_sha is what dump checks)
            asyncio.run(J.label([hpd], j, llm=OfferAccepts(), channels=("public",)))
            asyncio.run(jsp.label([spd], j, llm=SpStub()))
    return pairs


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("cc")
    return root, _world(root, ("fake:1.6:Let us all hold at 1.6", "fake:1.7:Let us all hold at 1.7", "fake:1.8"))


def test_dump_matches_the_registered_fingerprints_and_hides_names(world, tmp_path):
    root, pairs = world
    rep = O.dump(pairs, tmp_path, k=K)
    for part in ("a", "b"):
        assert rep[part]["fingerprint_check"] == {"match_both": rep[part]["prompts"]}
    # part a: the offer copies at k and the trunks; the silent model (m2) has nothing to label: empty records
    assert rep["a"]["items"] == 3 * (3 + 3) and rep["a"]["empty_no_message"] == 6
    assert rep["b"]["items"] == 3 * 2 * 2 and rep["b"]["empty_no_message"] == 4
    args = json.loads((tmp_path / "honeypot.json").read_text())
    assert set(args) == {"dir", "ids"} and len(args["ids"]) == rep["a"]["prompts"]
    assert all(i.startswith("a") for i in args["ids"]) and json.loads((tmp_path / "selfplay.json").read_text())["ids"]
    for f in (tmp_path / "prompts").iterdir():  # no run, model or folder name in an id or a prompt
        text = f.read_text()
        assert f.stem[1:].isdigit() and not any(x in text for x in ("fake:", "hp0", "hp1", "sp0", str(root)))
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    assert len(man["items"]) == rep["a"]["prompts"] + rep["b"]["prompts"]


def test_merge_writes_only_new_files_and_h1_runs_with_the_third_judge(world, tmp_path):
    root, pairs = world
    O.dump(pairs, tmp_path, k=K)
    before = _hashes(root)
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    prompts = {oid: (tmp_path / "prompts" / f"{oid}.txt").read_text() for oid in man["items"]}
    hp_ids = sorted(i for i in man["items"] if i.startswith("a"))
    # three offer copies (not trunks) of the talking models fail: their models must be withheld until the rerun
    bad, other, gone = [i for i in hp_ids if "/trunk/" not in man["items"][i]["item"]][:3]
    entries = [(oid, _stub_reply(t), O.EXPECTED_MODEL) for oid, t in sorted(prompts.items())
               if oid not in (bad, other, gone)]
    entries += [(bad, "I cannot label this.", O.EXPECTED_MODEL), (other, _stub_reply(prompts[other]), "claude-haiku-5-5"),
                (gone, None, O.EXPECTED_MODEL)]
    j1 = _journal(tmp_path / "wf1", entries)
    dry = O.merge(tmp_path, [j1])
    assert not dry["written"] and not list(root.rglob("*claude-opus-5-5-cc*"))  # a dry run writes nothing at all,
    assert not list(tmp_path.glob("rerun-*.json"))  # not even the rerun lists
    res = O.merge(tmp_path, [j1], write=True)
    pa = res["per_part"]["honeypot"]
    assert pa["unparsed"] == 1 and pa["other_model_or_no_transcript"] == 1 and pa["no_reply"] == 1
    assert pa["labelled"] == len(hp_ids) - 3
    assert pa["empty"] == 6 and res["per_part"]["selfplay"]["labelled"] == len([i for i in man["items"] if i[0] == "b"])
    assert sorted(json.loads((tmp_path / "rerun-honeypot-claude-opus-5-5-cc.json").read_text())["ids"]) == sorted([bad, other, gone])
    assert _hashes(root) == before  # the registered judges' files and the runs are untouched
    # unknown stays unknown: a model with an unlabelled offer copy is withheld, never ranked on part of its games
    solo = V.crossmodel_once(pairs, (CC,), K, n_boot=50, main_only=False)
    assert solo["left_out"] and solo["overridden"]
    # a second pass with the reruns: everything labelled, the earlier records superseded, nothing duplicated
    j2 = _journal(tmp_path / "wf2", [(i, _stub_reply(prompts[i]), O.EXPECTED_MODEL) for i in (bad, other, gone)])
    res2 = O.merge(tmp_path, [j1, j2], write=True)
    assert res2["rerun"] == {"honeypot": 0, "selfplay": 0} and res2["per_part"]["honeypot"]["labelled"] == 3
    assert O.merge(tmp_path, [j1, j2], write=True)["records"] == 0  # idempotent: labelled records are kept
    after = _hashes(root)  # (scoring.analyze, called by crossmodel_once, adds its own analysis.json / RESULTS.md)
    assert all(after.get(f) == h for f, h in before.items())
    # the third judge (stub replies = the registered stubs) gives the registered judges' verdicts
    for m, spec in pairs.items():
        h, sp = Path(spec["honeypot"]), Path(spec["selfplay"])
        reg = V.honeypot_games(h, (LUNA, HAIKU), K)["games"]
        assert {u: g["fused"] for u, g in V.honeypot_games(h, (CC,), K)["games"].items()} == \
               {u: g["fused"] for u, g in reg.items()}
        assert V.honeypot_games(h, (LUNA, CC), K)["games"] == reg
        assert V.selfplay_games(sp, (LUNA, CC))["games"] == V.selfplay_games(sp, (LUNA, HAIKU))["games"]
        assert jsp.window_agreement([sp], (CC,)) == jsp.window_agreement([sp], (LUNA,))
        assert J.agree(J.derive(J.current(h, CC)), J.derive(J.current(h, LUNA)))["tier"]["agreement"] == 1.0
    solo = V.crossmodel_once(pairs, (CC,), K, n_boot=50, main_only=False)
    pair = V.crossmodel_once(pairs, (LUNA, CC), K, n_boot=50, main_only=False)
    reg = V.crossmodel_once(pairs, (LUNA, HAIKU), K, n_boot=50, main_only=False)
    assert solo["left_out"] == [] and solo["ranks"] == reg["ranks"] == pair["ranks"]
    assert solo["decision"].startswith("OVERRIDDEN") and pair["decision"].startswith("OVERRIDDEN")
    assert all(r["honeypot"]["agreed_per_judge"] is not None and len(r["honeypot"]["agreed_per_judge"]) == 1
               for r in solo["models"].values())


def test_tries_are_capped_and_stale_prompts_are_never_written(tmp_path):
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6",))
    O.dump(pairs, tmp_path, k=K)
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    oid = sorted(i for i in man["items"] if i.startswith("a"))[0]
    js = [_journal(tmp_path / f"wf{n}", [(oid, "no", O.EXPECTED_MODEL)]) for n in range(J.RELABEL_PASSES)]
    O.merge(tmp_path, js[:1], write=True)
    assert oid in json.loads((tmp_path / "rerun-honeypot-claude-opus-5-5-cc.json").read_text())["ids"]
    O.merge(tmp_path, js, write=True)
    assert oid not in json.loads((tmp_path / "rerun-honeypot-claude-opus-5-5-cc.json").read_text())["ids"]  # 3 tries: stays unknown
    man["items"][oid]["prompt_sha"] = "0" * 16  # the item changed since dump
    (tmp_path / O.KEY_DIR / "manifest.json").write_text(json.dumps(man))
    assert O.merge(tmp_path, js)["per_part"]["honeypot"]["STALE"] == 1


def test_single_judge_shared_and_summary():
    a = {"msgs": [(14, 1), (12, 0)], "known": True}
    assert V.shared(a) == {"judges": ["yes"], "fused": "yes", "first_period": 12, "different_only": False}
    assert V.shared({"msgs": [], "known": True})["fused"] == "no" and V.shared(None)["fused"] is None
    side = {"planned": 2, "games": {"g1": {**V.shared(a), "first_round": 2}, "g2": V._unknown("x", 1)}}
    s = V.summary(side)
    assert s["withheld"] and s["agreed_per_judge"] == [1]


def test_dump_never_overwrites_and_a_fingerprint_gap_writes_nothing(tmp_path, monkeypatch):
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6",))
    O.dump(pairs, tmp_path / "d", k=K)
    with pytest.raises(SystemExit):  # an earlier dump is never overwritten
        O.dump(pairs, tmp_path / "d", k=K)
    monkeypatch.setattr(O, "registered_shas", lambda run, part: {j: {} for j in J.JUDGES})
    with pytest.raises(SystemExit):  # a prompt missing in a registered judge's records: hard error
        O.dump(pairs, tmp_path / "e", k=K)
    assert not (tmp_path / "e").exists()


def test_merge_binds_replies_to_the_dumped_file_and_the_results_own_agent(tmp_path):
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6",))
    O.dump(pairs, tmp_path, k=K)
    mp = tmp_path / O.KEY_DIR / "manifest.json"
    man = json.loads(mp.read_text())
    ids = sorted(i for i in man["items"] if i.startswith("a"))
    edited, swapped, orphan = ids[:3]
    reply = {i: _stub_reply((tmp_path / "prompts" / f"{i}.txt").read_text()) for i in ids}
    (tmp_path / "prompts" / f"{edited}.txt").write_text(" " + reply[edited])  # the file changed after dump
    j = _journal(tmp_path / "wf1", [(i, reply[i], O.EXPECTED_MODEL) for i in ids if i not in (swapped, orphan)]
                 + [(swapped, reply[swapped], O.EXPECTED_MODEL, "aother", "claude-haiku-5-5"),  # result by another agent
                    (orphan, reply[orphan], O.EXPECTED_MODEL, "aghost", None)])  # result agent has no transcript
    res = O.merge(tmp_path, [j])["per_part"]["honeypot"]
    assert res["STALE"] == 1 and res["other_model_or_no_transcript"] == 2 and res["labelled"] == len(ids) - 3
    # the one-off backfill for a manifest without file hashes: only files that still render to their prompt_sha
    for m in man["items"].values():
        m.pop("file_sha256")
    mp.write_text(json.dumps(man))
    assert O.merge(tmp_path, [j])["per_part"]["honeypot"]["STALE"] == len(ids)  # no hash: never used
    assert O.backfill_file_hashes(tmp_path) == {"backfilled": len(man["items"]) - 1, "not_backfilled": 1}
    assert O.merge(tmp_path, [j])["per_part"]["honeypot"]["STALE"] == 1


def test_append_refuses_an_unterminated_file_and_retries_are_counted_once(tmp_path):
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6",))
    O.dump(pairs, tmp_path, k=K)
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    ids = sorted(i for i in man["items"] if i.startswith("a"))
    oid, run = ids[0], Path(man["items"][ids[0]]["run"])
    good = _stub_reply((tmp_path / "prompts" / f"{oid}.txt").read_text())
    fails = [_journal(tmp_path / f"f{n}", [(oid, "no", O.EXPECTED_MODEL)]) for n in range(3)]
    late = _journal(tmp_path / "late", [(oid, good, O.EXPECTED_MODEL)])
    # the same journal given three times is ONE attempt: the item is still offered again
    r = O.merge(tmp_path, [fails[0]] * 3)
    assert r["journals_read"] == 1 and r["rerun"]["honeypot"] >= 1
    # three distinct failed attempts reach the cap: a fourth, successful one is ignored, never selected
    r = O.merge(tmp_path, [*fails, late])["per_part"]["honeypot"]
    assert r["attempts_over_cap_ignored"] == 1 and r.get("labelled", 0) == 0 and r["unparsed"] == 1
    # a success is never dropped by a later failed duplicate
    r = O.merge(tmp_path, [late, *fails])["per_part"]["honeypot"]
    assert r["labelled"] == 1
    # appending to an unterminated file is refused loudly, never repaired
    f = O.out_file(run, "a")
    f.write_text('{"id": "x"')
    with pytest.raises(SystemExit):
        O.merge(tmp_path, [late], write=True)
    assert f.read_text() == '{"id": "x"'


def test_judge_count_is_one_or_two():
    with pytest.raises(ValueError):
        V._check_judges(("a", "b", "c"))
    V._check_judges(("a",))
    s = V.summary({"planned": 0, "games": {}, "n_judges": 1})
    assert s["agreed_per_judge"] == [0] and s["withheld"]


def test_only_one_full_read_of_its_own_prompt_at_low_effort_is_accepted(tmp_path):
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6",))
    O.dump(pairs, tmp_path, k=K)
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    ids = sorted(i for i in man["items"] if i.startswith("a"))
    reply = {i: _stub_reply((tmp_path / "prompts" / f"{i}.txt").read_text()) for i in ids}
    var = dict(zip(ids, ("bash", "offset", "otherfile", "effort", "large")))
    j = _journal(tmp_path / "wf1", [(i, reply[i], O.EXPECTED_MODEL, None, None, var.get(i)) for i in ids])
    r = O.merge(tmp_path, [j], parts=("a",))
    assert r["per_part"]["honeypot"]["tool_violation"] == 3 and r["per_part"]["honeypot"]["wrong_effort"] == 1
    assert r["per_part"]["honeypot"]["labelled"] == len(ids) - 4  # allow_large reads the whole file: accepted
    assert r["rerun"] == {"honeypot": 4} and r["parts"] == ["honeypot"]
    # --part honeypot writes no self-play record, not even the empty ones
    O.merge(tmp_path, [j], write=True, parts=("a",))
    assert not list((tmp_path / "runs").rglob("words-sp-claude-opus-5-5-cc*"))
    assert list((tmp_path / "runs").rglob("words-claude-opus-5-5-cc*"))


def test_a_replica_name_writes_only_its_own_files_and_reads_as_an_override(tmp_path):
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6", "fake:1.7:Let us all hold at 1.7",
                                       "fake:1.8"))
    O.dump(pairs, tmp_path, k=K)
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    j = _journal(tmp_path / "wf1", [(i, _stub_reply((tmp_path / "prompts" / f"{i}.txt").read_text()),
                                     O.EXPECTED_MODEL) for i in sorted(man["items"])])
    O.merge(tmp_path, [j], write=True)
    main = {p: p.read_bytes() for p in (tmp_path / "runs").rglob("*claude-opus-5-5-cc.jsonl")}
    assert main and (tmp_path / "rerun-honeypot-claude-opus-5-5-cc.json").exists()
    r2 = "claude-opus-5-5-cc-r2"  # a low-effort replica name (the transcripts here are low effort)
    with pytest.raises(SystemExit):  # the same journal (the same agents) under a second name: refused
        O.merge(tmp_path, [j], judge=r2)
    assert {p: p.read_bytes() for p in (tmp_path / "runs").rglob("*claude-opus-5-5-cc.jsonl")} == main
    jr = _journal(tmp_path / "wfr2", [(i, _stub_reply((tmp_path / "prompts" / f"{i}.txt").read_text()),
                                       O.EXPECTED_MODEL) for i in sorted(man["items"])])  # its own agents
    res = O.merge(tmp_path, [jr], write=True, judge=r2)
    assert res["judge"] == r2 and res["per_part"]["honeypot"]["labelled"] == len([i for i in man["items"]
                                                                                  if i[0] == "a"])
    assert {p: p.read_bytes() for p in (tmp_path / "runs").rglob("*claude-opus-5-5-cc.jsonl")} == main
    with pytest.raises(SystemExit):  # and the reverse: the replica's agents never back the main name
        O.merge(tmp_path, [jr], judge=O.JUDGE_CC)
    assert len(list((tmp_path / "runs").rglob(f"words-{r2}.jsonl"))) == len(pairs)
    assert len(list((tmp_path / "runs").rglob(f"words-sp-{r2}.jsonl"))) == len(pairs)
    assert (tmp_path / f"rerun-selfplay-{r2}.json").exists()
    with pytest.raises(SystemExit):
        O.merge(tmp_path, [j], judge="claude-opus-5-5-cc-r4")
    # the analysis reads the replica by its name, alone or in a pair, always as an override
    hp0, sp0 = Path(pairs["m0"]["honeypot"]), Path(pairs["m0"]["selfplay"])
    assert V.honeypot_games(hp0, (r2,), K)["games"] == V.honeypot_games(hp0, (CC,), K)["games"]
    assert V.selfplay_games(sp0, (CC, r2))["games"] == V.selfplay_games(sp0, (CC, CC))["games"]
    assert jsp.window_agreement([sp0], (r2,)) == jsp.window_agreement([sp0], (CC,))
    assert J.agree(J.derive(J.current(hp0, r2)), J.derive(J.current(hp0, CC)))["tier"]["agreement"] == 1.0
    res = V.crossmodel_once(pairs, (LUNA, r2), K, n_boot=50, main_only=False)
    assert res["overridden"] and res["decision"].startswith("OVERRIDDEN")


def test_the_high_effort_run_has_its_own_name_effort_check_and_quick_check(tmp_path):
    from collusion.analysis import quickcheck as Q
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6", "fake:1.7:Let us all hold at 1.7"))
    runs = [Path(s["honeypot"]) for s in pairs.values()]
    O.dump(pairs, tmp_path, k=K)
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    ids = sorted(i for i in man["items"] if i.startswith("a"))
    reply = {i: _stub_reply((tmp_path / "prompts" / f"{i}.txt").read_text()) for i in ids}
    high, low_one = "claude-opus-5-5-cc-high", ids[0]
    assert O.JUDGE_EFFORT[high] == "high" and O.JUDGE_EFFORT[CC] == "low"
    j = _journal(tmp_path / "wfh", [(i, reply[i], O.EXPECTED_MODEL, None, None, "low" if i == low_one else "high")
                                    for i in ids])
    r = O.merge(tmp_path, [j], write=True, parts=("a",), judge=high)["per_part"]["honeypot"]
    assert r["labelled"] == len(ids) - 1 and r["wrong_effort"] == 1  # a low-effort transcript is no -high reply
    with pytest.raises(SystemExit):  # the -high run's agents can never back the main (low-effort) name
        O.merge(tmp_path, [j], parts=("a",))
    jl = _journal(tmp_path / "wfl", [(i, reply[i], O.EXPECTED_MODEL, None, None, "high") for i in ids])
    assert O.merge(tmp_path, [jl], parts=("a",))["per_part"]["honeypot"]["wrong_effort"] == len(ids)  # nor at high
    assert not list((tmp_path / "runs").rglob("words-claude-opus-5-5-cc.jsonl"))
    j2 = _journal(tmp_path / "wfh2", [(low_one, reply[low_one], O.EXPECTED_MODEL, None, None, "high")])
    assert O.merge(tmp_path, [j, j2], write=True, parts=("a",), judge=high)["per_part"]["honeypot"]["labelled"] == 1
    # the quick check reads the -high judge by name, alone, as an override (never the registered gate)
    text, key, page = Q.sheet(runs, (LUNA, HAIKU), K, sizes={"S1": 3, "S2": 0, "S3": 3}, seed=0)
    fused = {p["id"]: "yes" if key[p["id"]]["stratum"] == "S1" else "no" for p in page}
    out = ["# Quick check labels", "", f"Fingerprint: {key['_meta']['fingerprint']}", ""]
    for it in page:
        out += [f"## {it['id']}", "", "Labels:", f"Label: {fused[it['id']]}", ""]
    (tmp_path / "export.md").write_text("\n".join(out))
    res = Q.gate(tmp_path / "export.md", key, runs, (high,), K)
    assert "FINGERPRINT_MISMATCH" not in res and res["gate"]["registered"] is False and "OVERRIDDEN" in res
    assert set(res["per_judge_vs_hand"]) == {high} and res["fused_vs_hand"]["n"] == len(page)
    pool_h, _ = Q.message_calls(runs, (high,), K)
    pool_l, _ = Q.message_calls(runs, (LUNA,), K)
    assert {m: v["calls"][high] for m, v in pool_h.items()} == {m: v["calls"][LUNA] for m, v in pool_l.items()}


def test_merge_holds_one_lock_across_all_opus_names_and_names_the_journal_by_run_id(tmp_path):
    import fcntl
    import threading
    import time
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6",))
    O.dump(pairs, tmp_path, k=K)
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    ids = sorted(i for i in man["items"] if i.startswith("a"))
    j = _journal(tmp_path / "wf_x1", [(i, _stub_reply((tmp_path / "prompts" / f"{i}.txt").read_text()),
                                       O.EXPECTED_MODEL) for i in ids])
    done = []
    with open(tmp_path / "opus_cc.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)  # another merge holds the lock
        t = threading.Thread(target=lambda: done.append(O.merge(tmp_path, [j], write=True, parts=("a",),
                                                                 judge="claude-opus-5-5-cc-r2")))
        t.start()
        time.sleep(0.5)
        assert t.is_alive() and not done  # waits: the check and the write never interleave with another merge
        fcntl.flock(lk, fcntl.LOCK_UN)
    t.join(60)
    assert done and done[0]["per_part"]["honeypot"]["labelled"] == len(ids)
    recs = [json.loads(x) for f in (tmp_path / "runs").rglob("words-claude-opus-5-5-cc-r2.jsonl")
            for x in f.read_text().splitlines()]
    assert {r["cc"]["journal"] for r in recs if r.get("cc")} == {"wf_x1"}  # the run id only, no local path
    assert str(tmp_path) not in "".join(json.dumps(r.get("cc")) for r in recs)


def test_the_one_off_scrub_rewrites_only_the_journal_path(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "opus_cc_scrub_journal", Path(__file__).resolve().parents[1] / "scripts" / "opus_cc_scrub_journal.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6",))
    O.dump(pairs, tmp_path, k=K)
    man = json.loads((tmp_path / O.KEY_DIR / "manifest.json").read_text())
    j = _journal(tmp_path / "wf_ab12", [(i, _stub_reply((tmp_path / "prompts" / f"{i}.txt").read_text()),
                                         O.EXPECTED_MODEL) for i in sorted(man["items"])])
    O.merge(tmp_path, [j], write=True)
    clean = {f: f.read_bytes() for f in (tmp_path / "runs").rglob("*claude-opus-5-5-cc.jsonl")}
    for f, b in clean.items():  # as 4845262 wrote them: the absolute journal path
        f.write_bytes(b.replace(b'"journal": "wf_ab12"', f'"journal": "{j}"'.encode()))
    dirty = {f: f.read_bytes() for f in clean}
    assert dirty != clean
    res = S.scrub(tmp_path)
    n = sum(c.get("rewritten", 0) for c in res["per_judge"].values())
    assert not res["applied"] and n > 0 and {f: f.read_bytes() for f in clean} == dirty  # a dry run changes nothing
    S.scrub(tmp_path, apply=True)
    assert {f: f.read_bytes() for f in clean} == clean  # byte for byte what a run-id merge writes
    again = S.scrub(tmp_path)
    assert sum(c.get("rewritten", 0) for c in again["per_judge"].values()) == 0


def test_merges_from_different_out_dirs_serialize_on_the_run_folders(tmp_path):
    import fcntl
    import threading
    import time
    pairs = _world(tmp_path / "runs", ("fake:1.6:Let us all hold at 1.6",))
    O.dump(pairs, tmp_path / "d1", k=K)
    O.dump(pairs, tmp_path / "d2", k=K)  # a second dump folder over the SAME run files
    man = json.loads((tmp_path / "d2" / O.KEY_DIR / "manifest.json").read_text())
    ids = sorted(i for i in man["items"] if i.startswith("a"))
    j = _journal(tmp_path / "d2" / "wf_y1", [(i, _stub_reply((tmp_path / "d2" / "prompts" / f"{i}.txt").read_text()),
                                              O.EXPECTED_MODEL) for i in ids])
    run = Path(pairs["m0"]["honeypot"])
    done = []
    with open(run / ".opus_cc.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)  # a merge or scrub from d1 is writing this run folder
        t = threading.Thread(target=lambda: done.append(O.merge(tmp_path / "d2", [j], write=True, parts=("a",))))
        t.start()
        time.sleep(0.5)
        assert t.is_alive() and not done  # the merge from d2 waits for the run folder, not only for its own out-dir
        fcntl.flock(lk, fcntl.LOCK_UN)
    t.join(60)
    assert done and done[0]["per_part"]["honeypot"]["labelled"] == len(ids)


def test_lock_paths_are_resolved_before_sorting(tmp_path):
    import threading
    run = tmp_path / "runs" / "hp0"
    run.mkdir(parents=True)
    other = tmp_path / "runs" / "zz" / ".." / "hp0"  # the same folder, spelled differently
    (tmp_path / "out" / O.KEY_DIR).mkdir(parents=True)
    (tmp_path / "out" / O.KEY_DIR / "manifest.json").write_text(json.dumps({
        "items": {"a0001": {"run": str(run), "part": "a"}, "a0002": {"run": str(other), "part": "a"}},
        "empty": [{"run": str(run) + "/", "part": "a", "item": "x"}]}))
    (tmp_path / "runs" / "zz").mkdir()
    held = []

    def take():
        with O.run_locks(tmp_path / "out") as names:
            held.extend(names)
    t = threading.Thread(target=take)
    t.start()
    t.join(10)
    assert not t.is_alive()  # one lock per real folder: never taken twice by itself (no self-deadlock)
    assert len(held) == 2 and held[1] == os.path.join(os.path.realpath(run), ".opus_cc.lock")
