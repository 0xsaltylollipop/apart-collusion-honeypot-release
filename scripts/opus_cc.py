#!/usr/bin/env python3
"""A third words judge, Claude Opus 5.5, run through Claude Code subagents (the author, 25 Oct; EXPLORATORY, decided before
unblinding, never replacing the registered judges). Opus reads the byte-identical frozen judge prompt (judge.prompt /
judge_selfplay.sp_prompt, codebook v3, the same prompt fingerprint) of every item H1 uses, one subagent per item, and
its replies are parsed with the judge's own parser into records the analysis reads under the judge name
JUDGE_CC = "claude-opus-5-5-cc" (verdicts --judges claude-opus-5-5-cc, or gpt-6-luna,claude-opus-5-5-cc).

Parts (each its own Workflow run; ids are opaque, shuffled, and name neither the model nor the run):
  a  honeypot, public channel: the offer copies at the registered save point (k = 10) and the trunks of the k = 10
     units (H1's trunk-aware rule credits an assent to a trunk proposal)
  b  self-play: every 30-round window of the frozen pairs' self-play runs (judge_selfplay.sp_items)
The Workflow args of a part are honeypot.json (a) and selfplay.json (b).
An item with nothing to label (no tagged message) gets no prompt; merge writes its complete empty record, as
judge.label does.

    .venv/bin/python scripts/opus_cc.py dump --out <dir>      # prompts/<id>.txt, honeypot.json, selfplay.json, key
    .venv/bin/python scripts/opus_cc.py disagreements --out <dir> --a claude-opus-5-5-cc --b claude-opus-5-5-cc-high
        # reliability of two voters: pooled per-message kappa and counts only (writes nothing)
    .venv/bin/python scripts/opus_cc.py merge --out <dir> --part honeypot|selfplay|all <journal.jsonl> [...] [--write]
        [--judge-name claude-opus-5-5-cc-high|-high2|-r2|-r3]   # another run: own files, attempts, effort check

dump checks every rendered prompt against the prompt_sha recorded by BOTH registered judges for the same item (any
mismatch or gap: nothing is written), never overwrites an earlier dump, stores each prompt file's sha256, and prints
counts and sizes only. merge (dry run unless --write) parses each subagent's reply, checks from the subagent's
own transcript what it ran on and did (claude-opus-5-5 at the name's effort; one whole Read of its prompt), and
APPENDS records to the files of one Opus judge name only (<run>/words-<name>.jsonl, <run>/words-sp-<name>.jsonl;
--judge-name: JUDGE_CC, the voters claude-opus-5-5-cc-high and -high2 of opus-maj3, or a later replica -r2 / -r3). It
never opens another judge's file for writing; it refuses a journal whose agents already back records of another Opus
name (agent_registry reads every Opus judge file of the runs), and it holds the destination locks (run_locks: one per
run folder) from that check to the last append. Records name their journal by its workflow run id only (never a local path). Items left unlabelled (no
reply, unparsed, another model) are listed in rerun-<honeypot|selfplay>-<judge>.json for another pass, up to
judge.RELABEL_PASSES tries per item; after that they stay unknown, never "no"."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.analysis import crossmodel as cm  # noqa: E402
from collusion.analysis import judge as J  # noqa: E402
from collusion.analysis import judge_selfplay as JSP  # noqa: E402

JUDGE_CC = "claude-opus-5-5-cc"
# every Opus judge name and the effort its subagents must have run at (checked in each transcript): the main run at low
# effort, as the API judges; the two high-effort full runs (the author, 25 Oct: -high, -high2) at high; the three are the
# voters of opus-maj3 (majority.py); -r2 / -r3 kept for later low-effort replicas. Each name has its own files, rerun
# lists and 3-attempt cap, and one subagent backs the records of one name only (agent_registry).
JUDGE_EFFORT = {JUDGE_CC: "low", f"{JUDGE_CC}-high": "high", f"{JUDGE_CC}-high2": "high", f"{JUDGE_CC}-r2": "low",
                f"{JUDGE_CC}-r3": "low"}
JUDGE_NAMES = tuple(JUDGE_EFFORT)
EXPECTED_MODEL = "claude-opus-5-5"
REGISTERED_K = 10
PARTS = ("a", "b")
NAMES = {"a": "honeypot", "b": "selfplay"}  # the args file of each part
KEY_DIR = "key-do-not-open"


def select(pairs: dict, k: int = REGISTERED_K) -> dict[str, list[tuple[Path, dict, str | None]]]:
    """{part: [(run, item, rendered prompt or None)]} over the pairs, in a stable order (k: the save point, 10)."""
    out: dict[str, list] = {p: [] for p in PARTS}
    for model in sorted(m for m in pairs if not m.startswith("_")):
        hp = Path(pairs[model]["honeypot"])
        man = json.loads((hp / "manifest.json").read_text())
        trunks10 = {Path(info["trunk"]).name for info in man["units"].values() if int(info["k"]) == k}
        for it in J.items(hp, ("public",)):
            if it["kind"] == "trunk":
                part = "a" if it["unit"] in trunks10 else None
            else:
                part = "a" if it["step"] == "offer" and int(it["k"]) == k else None
            if part:
                out[part].append((hp, it, J.prompt(it)))
        sp = Path(pairs[model]["selfplay"])
        for it in JSP.sp_items(sp):
            out["b"].append((sp, it, JSP.sp_prompt(it)))
    return out


def registered_shas(run: Path, part: str) -> dict[str, dict[str, str | None]]:
    """{judge: {item id: prompt_sha}} of the registered judges' current-fingerprint records (load() keeps only records
    with this code's version and prompt hash, so a template change shows up as missing)."""
    res = {}
    for j in J.JUDGES:
        recs = (JSP.load(JSP.out_path(run, j), j) if part == "b" else J.load(J.out_paths(run, j)[0], j))
        res[j] = {r["id"]: r.get("prompt_sha") for r in recs}
    return res


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(pairs: dict, out: Path, seed: int = 0, k: int = REGISTERED_K) -> dict:
    """Write the prompts, the per-part Workflow args and the key (manifest). Refuses to overwrite an earlier dump (a
    reused opaque id would let an old journal label another item) and refuses outright when any rendered prompt
    differs from, or is missing in, a registered judge's records. Returns counts and sizes only (no ids, runs, text)."""
    prompts, key = out / "prompts", out / KEY_DIR
    if prompts.exists() or (key / "manifest.json").exists():
        raise SystemExit(f"{out}: an earlier dump exists (prompts/ or the manifest); never overwritten, use a new --out")
    sel = select(pairs, k)
    plan, report, bad = {}, {}, Counter()
    for part in PARTS:  # check everything before writing anything
        rows, shas, live = sel[part], {}, []
        for run, it, text in rows:
            if text is None:
                continue
            reg = shas.setdefault(run, registered_shas(run, part))
            got = [reg[j].get(it["id"]) for j in J.JUDGES]
            if any(g != J._sha(text) for g in got):
                bad[f"{part}: {'missing in a registered judge' if None in got else 'MISMATCH'}"] += 1
            live.append((run, it, text))
        random.Random(f"opus-cc:{part}:{seed}").shuffle(live)
        plan[part] = (rows, live)
    if bad:
        raise SystemExit(f"fingerprint check failed, nothing written: {dict(bad)}")
    prompts.mkdir(parents=True)
    key.mkdir(parents=True, exist_ok=True)
    manifest = {"judge": JUDGE_CC, "items": {}, "empty": [], "prompt_hash": J.PROMPT_HASH,
                "sp_prompt_hash": JSP.SP_PROMPT_HASH}
    for part, (rows, live) in plan.items():
        manifest["empty"] += [{"part": part, "run": str(run), "item": it["id"]} for run, it, t in rows if t is None]
        ids = []
        for n, (run, it, text) in enumerate(live, 1):
            oid, f = f"{part}{n:04d}", prompts / f"{part}{n:04d}.txt"
            f.write_text(text, encoding="utf-8")
            manifest["items"][oid] = {"part": part, "run": str(run), "item": it["id"], "prompt_sha": J._sha(text),
                                      "file_sha256": _file_sha(f)}
            ids.append(oid)
        (out / f"{NAMES[part]}.json").write_text(json.dumps({"dir": str(prompts), "ids": ids}), encoding="utf-8")
        sizes = [len(t) for _, _, t in live]
        lines = [max(len(x) for x in t.splitlines()) for _, _, t in live]
        report[part] = {"items": len(rows), "prompts": len(live), "empty_no_message": len(rows) - len(live),
                        "chars_median": int(st.median(sizes)) if sizes else None, "chars_max": max(sizes, default=None),
                        "longest_line_chars": max(lines, default=None), "fingerprint_check": {"match_both": len(live)}}
    (key / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return report


def backfill_file_hashes(out: Path) -> dict:
    """One-off (25 Oct): the first dump (34e1389) stored no per-file hash. Add each prompt file's sha256 from the files
    as they are now, only where the file still renders to the prompt_sha checked at dump time against both registered
    judges; any file that does not is reported and left without a hash (merge then never uses it)."""
    mp = out / KEY_DIR / "manifest.json"
    man = json.loads(mp.read_text())
    n, skipped = 0, 0
    for oid, m in man["items"].items():
        if m.get("file_sha256"):
            continue
        f = out / "prompts" / f"{oid}.txt"
        if f.exists() and J._sha(f.read_text(encoding="utf-8")) == m["prompt_sha"]:
            m["file_sha256"] = _file_sha(f)
            n += 1
        else:
            skipped += 1
    man.setdefault("notes", []).append(
        f"file_sha256 backfilled for {n} prompt files on 25 Oct from the files of the single dump of 34e1389 (each "
        f"re-checked against its prompt_sha); {skipped} not backfilled")
    mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return {"backfilled": n, "not_backfilled": skipped}


# ---------- merge ----------

_LINE_NO = re.compile(r"^\s*\d+\t", re.M)


def transcript(path: Path) -> dict | None:
    """What a subagent's own transcript proves (None: no transcript): the models of its real assistant turns
    (<synthetic> turns, written by the harness, ignored), the effort of every turn and API request, and its tool calls
    with, for each Read, the text it got back (the line-number prefixes of the Read tool removed)."""
    if not path.exists():
        return None
    models, efforts, calls, results = set(), set(), [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "api-request" and isinstance(r.get("params"), dict):
            efforts.add(((r["params"].get("output_config") or {}).get("effort")))
        m = r.get("message") if isinstance(r.get("message"), dict) else {}
        if r.get("type") == "assistant" and m.get("model") != "<synthetic>":
            models.add(m.get("model"))
            efforts.add(r.get("perTurnEffort") or r.get("effort"))
            for b in m.get("content") if isinstance(m.get("content"), list) else []:
                if b.get("type") == "tool_use":
                    calls.append({"id": b.get("id"), "name": b.get("name"), "input": b.get("input") or {}})
        if r.get("type") == "user":
            for b in m.get("content") if isinstance(m.get("content"), list) else []:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    text = c if isinstance(c, str) else "".join(x.get("text", "") for x in c or []
                                                                 if isinstance(x, dict))
                    results[b.get("tool_use_id")] = (text, bool(b.get("is_error")))
    for c in calls:
        c["result"] = results.get(c["id"])
    return {"models": sorted(models, key=str), "efforts": sorted(efforts, key=str), "calls": calls}


def check_transcript(tr: dict | None, prompt_file: Path, effort: str = "low") -> str | None:
    """None if the subagent's transcript shows exactly what the labelling allows; else the reason: every real turn on
    claude-opus-5-5, every turn and request at `effort` (the judge name's, JUDGE_EFFORT), and no tool call but ONE Read of exactly this prompt file
    (whole: no offset, limit or pages, only allow_large may be added; no error; and the text read back, line-number
    prefixes removed, equal to the file) plus StructuredOutput."""
    if tr is None or tr["models"] != [EXPECTED_MODEL]:
        return "model"
    if tr["efforts"] != [effort]:
        return "effort"
    reads = [c for c in tr["calls"] if c["name"] == "Read"]
    others = [c for c in tr["calls"] if c["name"] not in ("Read", "StructuredOutput")]
    if others or len(reads) != 1:
        return "tools"
    rd = reads[0]
    want = prompt_file.read_text(encoding="utf-8")
    if (not {"file_path"} <= set(rd["input"]) <= {"file_path", "allow_large"} or Path(str(rd["input"].get("file_path"))).resolve() != prompt_file.resolve()
            or rd["result"] is None or rd["result"][1] or _LINE_NO.sub("", rd["result"][0]).rstrip("\n")
            != want.rstrip("\n")):
        return "tools"
    return None


def read_journal(path: Path) -> list[dict]:
    """The attempts of one Workflow journal, in order: {"label" (opaque id), "agent_id", "reply", "transcript",
    "models"}. A result is joined to its label by key, but what it ran on and did comes from the transcript of the
    RESULT's own agentId (agent-<id>.jsonl next to the journal; none: unknown). A started agent without a result is an
    attempt with no reply."""
    started, results = {}, {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "started":
            started[r["key"]] = (r.get("label"), r.get("agentId"))
        elif r.get("type") == "result":
            results[r["key"]] = (r.get("agentId"), r.get("result"))
    out = []
    for key, (label, sid) in started.items():
        rid, res = results.get(key, (None, None))
        reply = res.get("reply") if isinstance(res, dict) and isinstance(res.get("reply"), str) else None
        aid = rid if key in results else sid
        tr = transcript(path.parent / f"agent-{aid}.jsonl") if aid else None
        out.append({"label": label, "agent_id": aid, "reply": reply, "transcript": tr,
                    "models": tr["models"] if tr else [], "journal": str(path)})
    return out


def _record(part: str, it: dict, text: str | None, labels, reply: str | None, attempts: int, cc: dict | None,
            judge: str = JUDGE_CC) -> dict:
    """A record in exactly the registered judges' format (judge.label / judge_selfplay.label), cost 0."""
    if part == "b":
        rec = {"id": it["id"], "channel": "public_sp", "run": it["run"], "game": it["game"], "first": it["first"],
               "last": it["last"], "judge": judge, "sp_version": JSP.SP_VERSION, "prompt_hash": JSP.SP_PROMPT_HASH,
               "prompt_sha": J._sha(text), "prompt_chars": len(text or ""), "labels": labels, "cost_usd": 0.0,
               "attempts": attempts, "messages": it["messages"]}
    else:
        rec = {**J._meta(it), "judge": judge, "version": J.VERSION, "prompt_hash": J.PROMPT_HASH, "labels": labels,
               "cost_usd": 0.0, "attempts": attempts, "prompt_sha": J._sha(text), "messages": it["messages"]}
    if text is None:
        rec["empty"] = True
    if reply is not None:
        rec["raw"] = reply
    if cc:
        rec["cc"] = cc
    return rec


def out_file(run: Path, part: str, judge: str = JUDGE_CC) -> Path:
    return JSP.out_path(run, judge) if part == "b" else J.out_paths(run, judge)[0]


def _existing(path: Path, part: str, judge: str = JUDGE_CC) -> dict[str, dict]:
    return {r["id"]: r for r in (JSP.load(path, judge) if part == "b" else J.load(path, judge))}


def _fresh(prev: dict | None, part: str, it: dict, text: str | None) -> bool:
    """A labelled record the analysis will still accept: judge_selfplay.fresh (bounds and prompt) for self-play,
    judge.current's test (prompt and item metadata) for the honeypot."""
    if prev is None or prev.get("labels") is None:
        return False
    if part == "b":
        return JSP.fresh(prev, it, text)
    return prev.get("prompt_sha") == J._sha(text) and all(prev.get(kk) == vv for kk, vv in J._meta(it).items())


def agent_registry(man: dict) -> dict[str, set[str]]:
    """{agent id: the Opus judge names whose records it backs}, over every Opus judge file (JUDGE_NAMES) of the
    manifest's runs: one subagent's transcript may back the records of one voter only."""
    reg: dict[str, set[str]] = defaultdict(set)
    places = {(m["run"], m["part"]) for m in man["items"].values()} | {(e["run"], e["part"]) for e in man["empty"]}
    for run, part in sorted(places):
        for name in JUDGE_NAMES:
            f = out_file(Path(run), part, name)
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    aid = (json.loads(line).get("cc") or {}).get("agent_id")
                except (ValueError, AttributeError):
                    continue
                if aid:
                    reg[aid].add(name)
    return reg


@contextmanager
def run_locks(out: Path):
    """The destination locks: <out>/opus_cc.lock, then one exclusive lock per run folder of the dump
    (<run>/.opus_cc.lock), taken in sorted order (no deadlock) and held to the end. Merges and scrubs from ANY out-dir
    that touch the same run files therefore serialize, for every Opus name."""
    import fcntl
    import os
    man = json.loads((out / KEY_DIR / "manifest.json").read_text(encoding="utf-8"))
    # resolved before sorting and deduplicating: /a and /z/../a are one lock, taken once, in one global order
    runs = sorted({os.path.realpath(r) for r in [m["run"] for m in man["items"].values()]
                   + [e["run"] for e in man["empty"]]})
    held = []
    try:
        for f in [out / "opus_cc.lock", *(Path(r) / ".opus_cc.lock" for r in runs)]:
            fh = open(f, "a")
            held.append(fh)
            fcntl.flock(fh, fcntl.LOCK_EX)
        yield [fh.name for fh in held]
    finally:
        for fh in reversed(held):
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()


def merge(out: Path, journals: list[Path], write: bool = False, parts: tuple[str, ...] = PARTS,
          judge: str = JUDGE_CC) -> dict:
    """_merge under the destination locks (run_locks), held from the voter-identity check to the last append, so two
    merges (from any out-dir, under any Opus name) can never both pass the check and then both write."""
    with run_locks(out):
        return _merge(out, journals, write, parts, judge)


def _merge(out: Path, journals: list[Path], write: bool = False, parts: tuple[str, ...] = PARTS,
           judge: str = JUDGE_CC) -> dict:
    """Parse the replies of the given parts into records of `judge` (JUDGE_CC, or a replica name of JUDGE_NAMES);
    with write=True append them to that judge's files only and write rerun-<honeypot|selfplay>-<judge>.json for those
    parts (a dry run writes nothing at all). Only the journals given are read, so each name has its own attempts and
    its own cap: pass only that run's journals (the main run's for JUDGE_CC, a replica's for its name). Journals are read
    in the order given (identical files once); per item only its first judge.RELABEL_PASSES distinct attempts
    (distinct agents) count, and the first of them whose transcript passes check_transcript and whose reply parses is
    used, so a later attempt never replaces an earlier success."""
    if judge not in JUDGE_NAMES:
        raise SystemExit(f"judge name {judge!r} is not one of {list(JUDGE_NAMES)}")
    man = json.loads((out / KEY_DIR / "manifest.json").read_text(encoding="utf-8"))
    if man.get("prompt_hash") != J.PROMPT_HASH or man.get("sp_prompt_hash") != JSP.SP_PROMPT_HASH:
        raise SystemExit("the judge prompts changed since dump: dump again into a new folder")
    seen_files, attempts = set(), defaultdict(list)
    for jp in journals:
        h = _file_sha(jp)
        if h in seen_files:
            continue  # the same journal given twice counts once
        seen_files.add(h)
        for e in read_journal(jp):
            if e["agent_id"] and e["agent_id"] in {x["agent_id"] for x in attempts[e["label"]]}:
                continue  # one agent is one attempt
            attempts[e["label"]].append(e)
    # voter identity: no agent of these journals may already back another Opus judge name's records (a journal merged
    # under two names, either way round, is refused before anything is counted or written)
    reg = agent_registry(man)
    clash = {e["agent_id"] for ts in attempts.values() for e in ts if reg.get(e["agent_id"], set()) - {judge}}
    if clash:
        others = sorted(set().union(*(reg[x] for x in clash)) - {judge})
        raise SystemExit(f"{len(clash)} agent(s) of these journals already back records of {others}: one subagent is "
                         f"one voter, never merged under {judge} as well")
    items_by_run: dict[str, dict[str, dict]] = {}

    def item(run: str, part: str, iid: str):
        if run not in items_by_run:
            p = Path(run)
            its = JSP.sp_items(p) if part == "b" else J.items(p, ("public",))
            items_by_run[run] = {x["id"]: x for x in its}
        return items_by_run[run].get(iid)

    new: dict[Path, list[dict]] = defaultdict(list)
    counts = {p: Counter() for p in PARTS}
    rerun = {p: [] for p in PARTS}
    have: dict[Path, dict] = {}
    for e in [x for x in man["empty"] if x["part"] in parts]:  # nothing to label: a complete empty record, as judge.label writes without a call
        part, it = e["part"], item(e["run"], e["part"], e["item"])
        if it is None or (JSP.sp_prompt(it) if part == "b" else J.prompt(it)) is not None:
            counts[part]["STALE_empty"] += 1
            continue
        f = out_file(Path(e["run"]), part, judge)
        if not _fresh(have.setdefault(f, _existing(f, part, judge)).get(it["id"]), part, it, None):
            new[f].append(_record(part, it, None, [], None, 0, None, judge))
        counts[part]["empty"] += 1
    for oid, m in sorted((o, x) for o, x in man["items"].items() if x["part"] in parts):
        part, it = m["part"], item(m["run"], m["part"], m["item"])
        text = None if it is None else (JSP.sp_prompt(it) if part == "b" else J.prompt(it))
        pf = out / "prompts" / f"{oid}.txt"
        # the reply answers the file the agent read: it must be the file hashed at dump time, rendering to the prompt
        # fingerprint-checked then, which the item still renders to now
        if (it is None or not m.get("file_sha256") or not pf.exists() or _file_sha(pf) != m["file_sha256"]
                or J._sha(pf.read_text(encoding="utf-8")) != m["prompt_sha"] or J._sha(text) != m["prompt_sha"]):
            counts[part]["STALE"] += 1  # never written
            continue
        f = out_file(Path(m["run"]), part, judge)
        if _fresh(have.setdefault(f, _existing(f, part, judge)).get(it["id"]), part, it, text):
            counts[part]["already_labelled"] += 1
            continue
        ts = attempts.get(oid) or []
        counts[part]["attempts_over_cap_ignored"] += max(0, len(ts) - J.RELABEL_PASSES)
        ts = ts[:J.RELABEL_PASSES]  # the cap, before any reply is chosen
        labels, used, why = None, None, set()
        for e in ts:  # the first counted attempt whose transcript passes and whose reply parses
            if e["reply"] is None:
                why.add("no_reply")  # started but returned nothing (skipped, or died)
                continue
            bad = check_transcript(e["transcript"], pf, JUDGE_EFFORT[judge])
            if bad:
                why.add(bad)
                continue
            got = J.parse_public(e["reply"], it["messages"]) if part == "b" else J.parse(it, e["reply"])
            if got is not None:
                labels, used = got, e
                break
            why.add("unparsed")
        if not ts:
            counts[part]["no_attempt_yet"] += 1
        elif labels is not None:
            counts[part]["labelled"] += 1
        else:  # one reason per item, the most serious first
            counts[part][next(n for w, n in (("tools", "tool_violation"), ("model", "other_model_or_no_transcript"),
                                              ("effort", "wrong_effort"), ("unparsed", "unparsed"),
                                              ("no_reply", "no_reply")) if w in why)] += 1
        if labels is None and len(ts) < J.RELABEL_PASSES:
            rerun[part].append(oid)
        if ts:  # a try was made: write what it gave (unlabelled records count as unknown, never "no")
            last = used or ts[-1]
            cc = {"opaque_id": oid, "agent_id": last["agent_id"], "models": last["models"], "tries": len(ts),
                  "journal": Path(last["journal"]).parent.name}  # the workflow run id, never a local path
            new[f].append(_record(part, it, text, labels, last["reply"], len(ts), cc, judge))
    if write:
        for f in new:
            # exactly the files of one of the three Opus names, and of the one being merged
            if judge not in JUDGE_NAMES or f.name not in (f"words-{judge}.jsonl", f"words-sp-{judge}.jsonl"):
                raise SystemExit(f"refusing to write {f}: not a file of {judge} (allowed: {list(JUDGE_NAMES)})")
            if f.exists() and f.stat().st_size and not f.read_bytes().endswith(b"\n"):
                raise SystemExit(f"refusing to append to {f}: its last line is unterminated (inspect it by hand)")
        for f, recs in new.items():
            with open(f, "a", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
        for part in parts:
            (out / f"rerun-{NAMES[part]}-{judge}.json").write_text(json.dumps({"dir": str(out / "prompts"),
                                                                     "ids": rerun[part]}), encoding="utf-8")
    return {"written": write, "judge": judge, "parts": [NAMES[p] for p in parts], "records": sum(len(v) for v in new.values()),
            "files": len(new), "journals_read": len(seen_files),
            "per_part": {NAMES[p]: dict(counts[p]) for p in parts}, "rerun": {NAMES[p]: len(rerun[p]) for p in parts}}


def disagreements(out: Path, a: str = JUDGE_CC, b: str = f"{JUDGE_CC}-high") -> dict:
    """A model-free reliability printout of two Opus voters (majority.py): over the items both read, the pooled
    per-message agreement and kappa of their H1 agreement-message sets (honeypot offer copies; self-play windows), and
    how many items they read differently or one of them did not read. Counts only: no ids, no per-model numbers, no
    file written (every voter is a full run; nothing is chosen from this)."""
    from collusion.analysis import majority as M
    from collusion.analysis import words as W
    man = json.loads((out / KEY_DIR / "manifest.json").read_text(encoding="utf-8"))
    places: dict[tuple[str, str], list[str]] = defaultdict(list)
    for m in man["items"].values():
        places[(m["run"], m["part"])].append(m["item"])
    stats = {p: {"items": 0, "differ": 0, "unread_by_either": 0, "x": [], "y": []} for p in PARTS}

    def tally(part, sa, sb, msgs):
        st_ = stats[part]
        st_["items"] += 1
        if sa is None or sb is None:
            st_["unread_by_either"] += 1
            return
        st_["differ"] += set(map(tuple, sa)) != set(map(tuple, sb))
        for pf in msgs:
            st_["x"].append("yes" if pf in set(map(tuple, sa)) else "no")
            st_["y"].append("yes" if pf in set(map(tuple, sb)) else "no")

    for (run_s, part), iids in sorted(places.items()):
        run = Path(run_s)
        if part == "a":
            rman = json.loads((run / "manifest.json").read_text())
            exp = W.expected_copies(run)
            vals = [W.copy_values(J.current(run, j), exp, rman) for j in (a, b)]
            its = {it["id"]: it for it in J.items(run, ("public",))}
            for iid in iids:
                it = its[iid]
                if it["kind"] != "copy":
                    continue  # H1's honeypot items are the offer copies (their trunk proposals count inside them)
                key = f"{it['run']}/{it['unit']}/{it['copy']}"
                sa, sb = (v[key]["agree_msgs"] if (v.get(key) or {}).get("labelled") else None for v in vals)
                tally(part, sa, sb, [(m["period"], m["firm"]) for m in it["messages"] if not m["context"]])
        else:
            sets = [M.window_sets(run, j) for j in (a, b)]
            its = {it["id"]: it for it in JSP.sp_items(run)}
            for iid in iids:
                tally(part, sets[0].get(iid), sets[1].get(iid),
                      [(m["period"], m["firm"]) for m in its[iid]["messages"] if not m["context"]])
    res = {"voters": [a, b]}
    for p_, st_ in stats.items():
        x, y = st_.pop("x"), st_.pop("y")
        res[NAMES[p_]] = {**st_, "messages_both_read": len(x), "kappa_per_message": J.kappa(x, y),
                          "raw_agreement": round(sum(i == j for i, j in zip(x, y)) / len(x), 4) if x else None}
    return res


def main():
    ap = argparse.ArgumentParser(prog="opus_cc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--pairs", default=str(cm.FROZEN_PAIRS))
    m = sub.add_parser("merge")
    m.add_argument("journals", nargs="+")
    m.add_argument("--out", required=True)
    m.add_argument("--part", required=True, choices=("honeypot", "selfplay", "all"),
                   help="which part to merge (honeypot first; self-play only once it is labelled)")
    m.add_argument("--judge-name", default=JUDGE_CC, choices=JUDGE_NAMES,
                   help="the main low-effort run (default), the high-effort run (-high), or a later replica (-r2, -r3): "
                        "each its own files, attempts and transcript effort check")
    m.add_argument("--write", action="store_true", help="append the records and write the rerun lists (default: "
                   "dry run, counts only, nothing written)")
    dg = sub.add_parser("disagreements", help="reliability of two Opus voters: pooled per-message kappa, counts only")
    dg.add_argument("--out", required=True)
    dg.add_argument("--a", default=JUDGE_CC, choices=JUDGE_NAMES)
    dg.add_argument("--b", default=f"{JUDGE_CC}-high", choices=JUDGE_NAMES)
    bf = sub.add_parser("backfill-hashes", help="one-off: add per-file sha256 to the first dump's manifest")
    bf.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "dump":
        print(json.dumps(dump(cm.load_pairs(a.pairs), Path(a.out)), indent=2))
    elif a.cmd == "disagreements":
        print(json.dumps(disagreements(Path(a.out), a.a, a.b), indent=2))
    elif a.cmd == "backfill-hashes":
        print(json.dumps(backfill_file_hashes(Path(a.out)), indent=2))
    else:
        parts = PARTS if a.part == "all" else tuple(p for p in PARTS if NAMES[p] == a.part)
        print(json.dumps(merge(Path(a.out), [Path(j) for j in a.journals], a.write, parts, a.judge_name), indent=2))


if __name__ == "__main__":
    main()
