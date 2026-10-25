#!/usr/bin/env python3
"""Blind progress of honeypot runs: counts, failures, cost and time only, never a price, message or measure
(PREREG A: while the main runs go we watch progress, errors and costs, not results).

    .venv/bin/python scripts/progress.py runs/hp-main-claude-haiku-5-5 [more run dirs ...]
    .venv/bin/python scripts/progress.py --all            # every runs/hp-main-* under HP_MAIN_ROOT (or this checkout)

Works for laptop runs and for pod runs fetched by scripts/pod/pod.py (their pod.json adds the pod's phase and cost).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.analysis.report import costs  # noqa: E402  (cost fields only)
from collusion.experiments.common import _finished, _last_period  # noqa: E402
from collusion.experiments.honeypot import _debrief_complete  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main-honeypot.json"
MAIN = Path(os.environ.get("HP_MAIN_ROOT") or json.loads(CONFIG.read_text())["main_root"])


def planned_copies(man: dict) -> list[tuple[Path, int]]:
    """(copy dir, last round) of every copy the manifest's design plans for the units registered so far."""
    out = []
    reps, steps = man.get("reps", 1), man.get("steps") or []
    pc_steps = man.get("pc_steps") or [] if man.get("pc_persona") else []
    for unit, u in (man.get("units") or {}).items():
        if not u.get("headroom"):
            continue
        k = u["k"]
        names = [f"{st}-b{b}" for st in steps for b in range(reps)]
        if k in (man.get("ks") or []):
            names += [f"pc-{st}-b{b}" for st in pc_steps for b in range(reps)]
        out += [(Path(man["_dir"]) / "copies" / unit / n, k + man["copy_len"]) for n in names]
    return out


def progress(d: Path) -> dict:
    mpath = d / "manifest.json"
    if not mpath.exists():
        return {"run": d.name, "status": "no manifest yet"}
    man = json.loads(mpath.read_text())
    man["_dir"] = str(d)
    ks = man.get("ks") or [10]
    seeds = [man["seed0"] + g for g in range(man["games"])]
    trunks_done = sum(_last_period(d / "trunks" / f"trunk-s{s}") >= ks[-1] for s in seeds)
    plan = planned_copies(man)
    started = sum((cd / "rounds.jsonl").exists() for cd, _ in plan)
    finished = [cd for cd, until in plan if _finished(cd, until)]
    errors = 0
    for cd, _ in plan:
        s = cd / "summary.json"
        if s.exists():
            try:
                errors += json.loads(s.read_text()).get("error") is not None
            except Exception:
                errors += 1
    debrief = sum(_debrief_complete(cd) for cd in finished) if man.get("debrief") else None
    c = costs(d)
    t0 = man.get("started")
    elapsed_h = None
    if t0:
        elapsed_h = round((datetime.now(timezone.utc) - datetime.fromisoformat(t0)).total_seconds() / 3600, 2)
    gave_up = 0
    if man.get("debrief"):
        from collusion.game.engine import read_jsonl
        gave_up = sum(sum(1 for r in read_jsonl(cd / "debrief.jsonl") if r.get("gave_up")) for cd in finished)
    rep = {"run": d.name, "model": man.get("model"), "status": man.get("status", "running or old manifest"),
           "incomplete": man.get("incomplete"),
           "trunks": f"{trunks_done}/{len(seeds)}",
           "units_registered": len(man.get("units") or {}),
           "units_without_headroom": sum(not u.get("headroom") for u in (man.get("units") or {}).values()),
           "copies": f"{len(finished)}/{len(plan)} finished, {started} started, {errors} with an error",
           "debriefs_complete": None if debrief is None else f"{debrief}/{len(finished)}",
           "debrief_subjects_gave_up": gave_up,  # counted as done, but no usable answer after 3 tries
           "cost_usd": round(c["subjects"] + c["c"] + c["debrief"], 2),  # API spend of this run only
           "calls": c["calls"], "elapsed_h": elapsed_h}
    sens = d / "_pod" / "sens"  # a pod's sensitivity rows (their own runs; never in this run's cost)
    if sens.is_dir():
        rep["sensitivity_rows"] = {s.name: progress(s) for s in sorted(sens.iterdir()) if s.is_dir()}
    pod = d / "pod.json"
    if pod.exists():
        p = json.loads(pod.read_text())
        end = p.get("ended") or time.time()
        rep["pod"] = {"phase": p.get("phase"), "gpu_cost_usd_incl_sens": round((p.get("cost_per_hr") or 0) * (end - p["created"])
                                                                  / 3600, 2),
                      "budget_usd": p.get("budget_usd"), "last_error": p.get("last_error")}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every runs/hp-main-* run")
    a = ap.parse_args()
    dirs = [Path(x) for x in a.dirs]
    if a.all:
        dirs += sorted(p for p in (MAIN / "runs").glob("hp-main-*") if p.is_dir())
    if not dirs:
        ap.error("give run dirs or --all")
    for d in dirs:
        print(json.dumps(progress(d)))


if __name__ == "__main__":
    main()
