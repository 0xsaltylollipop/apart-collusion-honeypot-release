"""Verification of the H2 selfish baseline (the author 24 Oct): the myopic best response it is built on, and the closed-loop
replay (scoring.selfish_replay), checked against independent computations. Offline, synthetic and robot data only.

  (i)   fixed point: the best response to symmetric Nash prices is the Nash price
  (ii)  optimum: the best response equals a brute-force argmax of the logit profit on a 1e-5 grid over
        [marginal cost, ceiling], for random rival prices incl. near the cap and below cost
  (iii) strategic complements: the best response never falls when a rival's price rises
  (iv)  the replay reproduces the competitor robot's actual prices round by round in every copy of a robot run
        (scripts/robot_controls.py --out <dir>; pass that dir), so a selfish robot scores exactly 0 on H2
  (v)   price units: the replay's best response is unrounded (as the competitor robot and gap_closed); the effect of
        rounding it to cents, as an LLM's reply is, on the baseline of every robot copy

    .venv/bin/python scripts/verify_selfish_baseline.py [--robots <robot_controls out dir>] [--cases 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.analysis.scoring import (BENCH, DIST_BOUNDS, MARKET, SUBJECTS, distance_closed,  # noqa: E402
                                        selfish_replay)
from collusion.experiments.common import _rounds  # noqa: E402

CEILING = 2.8873621120069277  # the stated ceiling of the 3-firm game (copy metadata "ceiling"; (iv) lists it)


def grid_br(i: int, p, upper: float, step: float = 1e-5) -> float:
    """Brute-force argmax of firm i's logit profit over [alpha*c_i, upper] on a grid, vectorised, from the market's
    own parameters (not its best_response)."""
    m = MARKET
    lo = m.alpha * m.c[i]
    xs = np.arange(lo, upper + step / 2, step)
    a = np.asarray(m.a, dtype=float)
    others = sum(math.exp((a[j] - p[j] / m.alpha) / m.mu) for j in range(m.n) if j != i) + math.exp(m.a0 / m.mu)
    own = np.exp((a[i] - xs / m.alpha) / m.mu)
    prof = (xs - lo) * m.beta * own / (own + others)
    return float(xs[int(np.argmax(prof))])


def check_fixed_point() -> dict:
    pn = BENCH["nash_prices"]
    errs = [abs(MARKET.best_response(i, list(pn), upper=CEILING) - pn[i]) for i in range(MARKET.n)]
    return {"nash": pn[0], "max_abs_error": max(errs), "pass": max(errs) < 1e-6}


def check_grid(cases: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    fixed = [[2.887, 2.887], [2.88, 2.85], [0.5, 0.9], [0.9, 2.887], [1.4724, 1.4724], [2.84, 2.84], [1.0, 1.0]]
    rivals = fixed + [[rng.uniform(0.5, CEILING), rng.uniform(0.5, CEILING)] for _ in range(cases)]
    worst, worst_case, at_ceiling = 0.0, None, 0
    for r in rivals:
        p = [0.0, r[0], r[1]]
        br = MARKET.best_response(0, list(p), upper=CEILING)
        g = grid_br(0, p, CEILING)
        at_ceiling += abs(br - CEILING) < 1e-6
        if abs(br - g) > worst:
            worst, worst_case = abs(br - g), {"rivals": r, "best_response": br, "grid": g}
    return {"cases": len(rivals), "max_abs_diff": worst, "worst_case": worst_case, "at_ceiling": at_ceiling,
            "pass": worst < 1e-4}


def check_monotone(cases: int, seed: int = 1) -> dict:
    rng = random.Random(seed)
    viol, worst = 0, 0.0
    for _ in range(cases):
        p = [0.0, rng.uniform(0.5, CEILING), rng.uniform(0.5, CEILING)]
        q = list(p)
        j = rng.choice([1, 2])
        q[j] = min(CEILING, q[j] + rng.uniform(0.001, 0.5))
        d = MARKET.best_response(0, q, upper=CEILING) - MARKET.best_response(0, p, upper=CEILING)
        if d < -1e-9:
            viol += 1
            worst = min(worst, d)
    return {"cases": cases, "violations": viol, "worst_drop": worst, "pass": viol == 0}


def check_replay(robots: Path) -> dict:
    """(iv) and (v) on every copy of the competitor robot run, and (v) on all robot runs."""
    out = {"copies": 0, "max_abs_price_diff": 0.0, "ceilings": set(), "rounding_max_baseline_change": 0.0}
    for man in sorted(robots.glob("*/manifest.json")):
        run = man.parent
        mf = json.loads(man.read_text())
        k, hold = mf["ks"][0], mf.get("constants", {}).get("hold", 12)
        for cd in sorted((run / "copies").glob("*/*")):
            meta = json.loads((cd / "metadata.json").read_text())
            out["ceilings"].add(meta["ceiling"])
            rep = selfish_replay(cd, k, hold)
            rs = {r["period"]: r for r in _rounds(cd)}
            base = [rs[k]["prices"][i] for i in SUBJECTS]
            X = (meta["parent"]["interventions"][0]["spec"].get("trap") or {}).get("X")
            if X is None:  # the no-trap copy: X is the unit's (manifest) price C would have set
                X = mf["units"][cd.parent.name]["X"]
            if run.name == "competitor":
                out["copies"] += 1
                gone = [t for t in rep if t not in rs]  # an incomplete copy is a failure, never compared on part
                if gone:
                    out.setdefault("incomplete_copies", []).append(f"{cd.parent.name}/{cd.name}: no rounds {gone}")
                    continue
                diff = max(abs(rep[t][i] - rs[t]["prices"][i]) for t in rep for i in (*SUBJECTS, 2))
                out["max_abs_price_diff"] = max(out["max_abs_price_diff"], diff)
            window = [t for t in range(k + 2, k + hold + 1)]
            def sb(round_cents):
                v = [distance_closed(round(rep[t][i], 2) if round_cents else rep[t][i], base[n], X, False,
                                     DIST_BOUNDS) for t in window for n, i in enumerate(SUBJECTS)]
                v = [x for x in v if x is not None]
                return sum(v) / len(v) if v else None
            a, b = sb(False), sb(True)
            if a is not None and b is not None:
                out["rounding_max_baseline_change"] = max(out["rounding_max_baseline_change"], abs(a - b))
    out["ceilings"] = sorted(out["ceilings"])
    # every planned copy of the competitor run: scored units x steps (5 steps x reps, no positive control)
    cm = json.loads((robots / "competitor" / "manifest.json").read_text()) if (robots / "competitor").exists() else {}
    scored = [u for u, i in (cm.get("units") or {}).items() if i.get("headroom")]
    out["copies_planned"] = len(scored) * len(cm.get("steps") or []) * int(cm.get("reps") or 1)
    out["pass"] = (out["copies"] > 0 and out["copies"] == out["copies_planned"] and not out.get("incomplete_copies")
                   and out["max_abs_price_diff"] < 1e-9)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--robots", type=Path, default=None)
    ap.add_argument("--cases", type=int, default=2000)
    a = ap.parse_args(argv)
    res = {"fixed_point": check_fixed_point(), "grid": check_grid(a.cases), "monotone": check_monotone(a.cases)}
    if a.robots:
        res["replay"] = check_replay(a.robots)
    print(json.dumps(res, indent=2, default=str))
    ok = all(v["pass"] for v in res.values())
    print(f"\npass: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
