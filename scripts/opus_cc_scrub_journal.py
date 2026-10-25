#!/usr/bin/env python3
"""One-off (25 Oct): records merged with 4845262 name their workflow journal by its absolute local path (cc.journal),
and those label files are released. Rewrite ONLY that field, to the workflow run id (the journal's folder, wf_...);
every other byte of every line stays as it is (a textual replacement of that one JSON string, then checked: the line
parses to the same record with only cc.journal changed). Every Opus judge file of the dump's runs is read
(opus_cc.JUDGE_NAMES, honeypot and self-play). Dry run by default: counts only. --apply rewrites each file that needs
it (a new file next to it, fsynced, then an atomic replace, the folder fsynced before and after) while holding the
same destination locks as merge (opus_cc.run_locks), so no merge from any out-dir runs at the same time.

    .venv/bin/python scripts/opus_cc_scrub_journal.py --out <scratchpad>/opusjudge [--apply]"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location("opus_cc", Path(__file__).resolve().parent / "opus_cc.py")
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)


def scrub_line(line: str) -> tuple[str, str]:
    """(the line, what happened): "rewritten", "already_run_id", "no_journal"."""
    rec = json.loads(line)
    j = (rec.get("cc") or {}).get("journal")
    if not j:
        return line, "no_journal"
    if "/" not in j:
        return line, "already_run_id"
    run_id = Path(j).parent.name
    if not run_id.startswith("wf_"):
        raise SystemExit(f"unexpected journal folder {run_id!r}: nothing rewritten")
    old, new = '"journal": ' + json.dumps(j), '"journal": ' + json.dumps(run_id)
    if line.count(old) != 1:
        raise SystemExit("unexpected record layout (cc.journal not found exactly once): nothing rewritten")
    out = line.replace(old, new)
    want = json.loads(line)
    want["cc"]["journal"] = run_id
    if json.loads(out) != want:
        raise SystemExit("the rewrite changed more than cc.journal: nothing rewritten")
    return out, "rewritten"


def files(out: Path) -> list[tuple[str, Path]]:
    man = json.loads((out / O.KEY_DIR / "manifest.json").read_text(encoding="utf-8"))
    places = {(m["run"], m["part"]) for m in man["items"].values()} | {(e["run"], e["part"]) for e in man["empty"]}
    res = []
    for run, part in sorted(places):
        for name in O.JUDGE_NAMES:
            f = O.out_file(Path(run), part, name)
            if f.exists():
                res.append((name, f))
    return res


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def scrub(out: Path, apply: bool = False) -> dict:
    with O.run_locks(out):  # the same destination locks as merge: no merge from any out-dir runs meanwhile
        counts: dict[str, Counter] = {}
        plan = []
        for name, f in files(out):
            c = counts.setdefault(name, Counter())
            c["files"] += 1
            text = f.read_text(encoding="utf-8")
            lines = text.split("\n")
            new = []
            for line in lines:
                if not line.strip():
                    new.append(line)
                    continue
                nl, what = scrub_line(line)
                c["records"] += 1
                c[what] += 1
                new.append(nl)
            if new != lines:
                c["files_to_rewrite"] += 1
                plan.append((f, "\n".join(new)))
        if apply:
            for f, text in plan:
                tmp = f.with_name(f.name + ".scrub-tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(text)
                    fh.flush()
                    os.fsync(fh.fileno())  # the new content is on disk before it replaces the old
                _fsync_dir(f.parent)
                os.replace(tmp, f)
                _fsync_dir(f.parent)  # and the rename too
        return {"applied": apply, "per_judge": {n: dict(c) for n, c in counts.items()}}


def main():
    ap = argparse.ArgumentParser(prog="opus_cc_scrub_journal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true", help="rewrite (default: dry run, counts only)")
    a = ap.parse_args()
    print(json.dumps(scrub(Path(a.out), a.apply), indent=2))


if __name__ == "__main__":
    main()
