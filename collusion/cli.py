"""Command line: python -m collusion {run, analyze, robots, selfplay, selfplay-stats, crossmodel}."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from importlib.util import find_spec
from pathlib import Path

from collusion.analysis.report import render
from collusion.analysis.scoring import CANDIDATES, analyze, selfplay_stats
from collusion.experiments.honeypot import C_CHECK_MODEL, FIXED_C_MODEL, HOLD, PERSONAS, STEP_FRAC, STEPS, run
from collusion.experiments.selfplay import selfplay

# laptop-only modules, left out of the pod's code bundle (scripts/pod/pod.py LAPTOP_ONLY): imported only when present,
# so the pod's entry points (run, selfplay, selfplay-stats, analyze) work without them
HAS_CROSSMODEL = find_spec("collusion.analysis.crossmodel") is not None
HAS_ROBOTS = find_spec("collusion.experiments.robots") is not None
if HAS_CROSSMODEL:
    from collusion.analysis.crossmodel import CI_FIELDS, PRIMARY_CI, run_crossmodel
if HAS_ROBOTS:
    from collusion.experiments.robots import robot_check


def main():
    ap = argparse.ArgumentParser(prog="collusion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--games", type=int, default=2)
    r.add_argument("--ks", default="10")
    r.add_argument("--copy-len", type=int, default=30)
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--steps", default=",".join(STEPS))
    r.add_argument("--seed0", type=int, default=1)
    r.add_argument("--c-model", default=FIXED_C_MODEL)
    r.add_argument("--c-effort", default="low")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--sampling", default="{}", help="JSON: reasoning_effort, max_tokens, temperature, ...")
    r.add_argument("--tag", default="pre-sprint-trap")
    r.add_argument("--private", action="store_true", help="subjects have a private line to C only (no A-B chat)")
    r.add_argument("--hold", type=int, default=HOLD, help="rounds C holds X (sensitivity: 6 / 12 / 20)")
    r.add_argument("--persona", default=None, choices=sorted(PERSONAS),
                   help="old positive control: an instruction in the subjects' system prompt from round 1")
    r.add_argument("--pc-persona", default=None, choices=sorted(PERSONAS),
                   help="positive control (24 Oct): extra copies where the subjects get this instruction from k+1 only")
    r.add_argument("--pc-steps", default="none,offer", help="steps that get a positive-control copy")
    r.add_argument("--c-check", default=C_CHECK_MODEL,
                   help="pre-send checker of C's live trunk messages ('' = the legacy regex guard and live C after k)")
    r.add_argument("--c-script-lead", type=int, default=0,
                   help="start C's scripted style this many rounds before the first save point (default off)")
    r.add_argument("--primary-k", type=int, default=None, help="primary save point (default 10 if run, else first)")
    r.add_argument("--h1", default=None, choices=sorted(CANDIDATES),
                   help="the pre-registered price predictor, the paper's H2 (stored in the manifest as \"h1\")")
    r.add_argument("--debrief", action=argparse.BooleanOptionalAction, default=None,
                   help="suspicion probe after each copy (default: on for v3 runs of a model)")
    r.add_argument("--sub-ks", default="", help="extra save points in the first --sub-games games only (e.g. 5)")
    r.add_argument("--sub-games", type=int, default=0, help="how many games (from --seed0) get the --sub-ks save points")
    r.add_argument("--step-frac", type=float, default=STEP_FRAC, help="C's raise (sensitivity subset: 1.08 / 1.20)")
    r.add_argument("--c-normal", default="follow", choices=("follow", "best_response"),
                   help="C's normal rule (sensitivity subset: best_response)")
    r.add_argument("--max-usd", type=float, default=None,
                   help="spend stop for this run (subjects + C + debrief, from the logs): nothing new starts above it")
    r.add_argument("--frozen-json", default=None,
                   help="main runs: JSON of the frozen design (configs/main-honeypot.json via scripts/launch_main.py); "
                        "the run refuses to start if any named key differs (start gate)")
    r.add_argument("--trunks-from", default=None,
                   help="sensitivity subset: fork from this finished run's trunks (paired with its units)")
    r.add_argument("--blind", action="store_true",
                   help="main runs: print progress and status only, no unit prices and no analysis (run `analyze` "
                        "after unblinding)")
    r.add_argument("--unblind", action="store_true",
                   help="resume a blind run with normal output (only after the analysis decisions are fixed)")
    sp = sub.add_parser("selfplay", help="answer key: self-play of the tested model (2 or 3 firms), no firm C")
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--games", type=int, default=3)
    sp.add_argument("--periods", type=int, default=300)
    sp.add_argument("--seed0", type=int, default=1)
    sp.add_argument("--chat", action="store_true", help="with chat (default: no chat)")
    sp.add_argument("--firms", type=int, default=3, choices=(2, 3))
    sp.add_argument("--concurrency", type=int, default=16)
    sp.add_argument("--sampling", default="{}")
    sp.add_argument("--tag", default="pre-sprint-selfplay")
    sps = sub.add_parser("selfplay-stats")
    sps.add_argument("out")
    sps.add_argument("--upto", type=int, default=None, help="score as if the games had stopped at this round")
    sps.add_argument("--threshold", type=float, default=None,
                     help="pre-registered Delta for a 'high-Delta' game (k of n; a convention, not a collusion test)")
    a = sub.add_parser("analyze")
    a.add_argument("out")
    a.add_argument("--unblind", action="store_true", help="analyse a blind main run (after the analysis is frozen)")
    if HAS_CROSSMODEL:
        cm = sub.add_parser("crossmodel", help="prices across models: S against every price candidate (H2), H3, H4")
        cm.add_argument("--pairs", default=None,
                        help='JSON: {model: {"selfplay": dir, "honeypot": dir, "family": str}} (default: the frozen '
                             'configs/pairs.json; main runs must match it)')
        cm.add_argument("--out", required=True, help="folder for crossmodel.json and crossmodel.md")
        cm.add_argument("--h2", "--h1", dest="h2", default=None, choices=sorted(CANDIDATES),
                        help="override the registered H2 (prices) predictor (scoring.H2_PREDICTOR, the default); "
                             "an override is reported loudly and is never a registered decision; --h1 is its old name")
        cm.add_argument("--boot", type=int, default=2000, help="bootstrap draws")
        cm.add_argument("--primary-ci", default=None, choices=sorted(CI_FIELDS),
                        help=f"override the registered primary CI (crossmodel.PRIMARY_CI = {PRIMARY_CI!r}); "
                             "an override is reported loudly in the output")
        cm.add_argument("--allow-nonmain", action="store_true", help="test data only: admit non hp-main-* runs")
        cm.add_argument("--unblind", action="store_true",
                        help="required: crossmodel analyses (and writes analysis.json / RESULTS.md into) every run "
                             "and reveals S; only after the analysis is frozen")
    if HAS_ROBOTS:
        b = sub.add_parser("robots")
        b.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        frozen = json.loads(args.frozen_json) if args.frozen_json else None
        if frozen is not None and frozen.pop("blind", False) and not args.blind:  # Codex 24 Oct: blindness is frozen too
            raise SystemExit("start gate: the frozen design is a blind run; pass --blind")
        man = asyncio.run(run(args.model, Path(args.out), games=args.games,
                              ks=[int(x) for x in args.ks.split(",")], copy_len=args.copy_len, reps=args.reps,
                              steps=args.steps.split(","), sampling=json.loads(args.sampling),
                              c_model=args.c_model or None, c_effort=args.c_effort or None, seed0=args.seed0,
                              concurrency=args.concurrency, tag=args.tag, private=args.private, hold=args.hold,
                              persona=args.persona, pc_persona=args.pc_persona,
                              pc_steps=args.pc_steps.split(",") if args.pc_persona else None,
                              c_check=args.c_check or None, c_script_lead=args.c_script_lead,
                              primary_k=args.primary_k, h1=args.h1, debrief=args.debrief,
                              sub_ks=[int(x) for x in args.sub_ks.split(",") if x], sub_games=args.sub_games,
                              step_frac=args.step_frac, c_normal=args.c_normal, max_usd=args.max_usd,
                              frozen=frozen, blind=args.blind, unblind=args.unblind,
                              trunks_from=Path(args.trunks_from) if args.trunks_from else None))
        if man.get("blind"):  # counts only (Codex 24 Oct: the routine output must not reveal results before unblinding)
            print(json.dumps({"status": man.get("status"), "incomplete": man.get("incomplete"),
                              "units": len(man["units"]),
                              "units_without_headroom": sum(not i["headroom"] for i in man["units"].values())}))
        else:
            print(json.dumps({u: {kk: v for kk, v in i.items() if kk in ("X", "headroom")} for u, i in
                              man["units"].items()}))
            print(render(analyze(Path(args.out))))
        if man.get("status") != "complete":
            sys.exit(3)  # incomplete play or debrief: a retry loop or the pod runner resumes the run
    elif args.cmd == "selfplay":
        asyncio.run(selfplay(args.model, Path(args.out), games=args.games, periods=args.periods, seed0=args.seed0,
                             sampling=json.loads(args.sampling), chat=args.chat, concurrency=args.concurrency,
                             firms=args.firms, tag=args.tag))
        print(json.dumps(selfplay_stats(Path(args.out)), indent=2))
    elif args.cmd == "selfplay-stats":
        print(json.dumps(selfplay_stats(Path(args.out), upto=args.upto, threshold=args.threshold), indent=2))
    elif args.cmd == "analyze":
        mp = Path(args.out) / "manifest.json"
        if mp.exists() and json.loads(mp.read_text()).get("blind") and not args.unblind:
            raise SystemExit(f"{args.out} is a blind run: analyse it with --unblind, only after the analysis "
                             "decisions (PREREG Section B) are fixed")
        print(render(analyze(Path(args.out))))
    elif args.cmd == "crossmodel":
        if not args.unblind:  # as analyze and verdicts (fresh freeze review 24 Oct)
            raise SystemExit("crossmodel analyses every run and reveals S: pass --unblind, only after the analysis "
                             "decisions (PREREG Section B) are fixed")
        from collusion.analysis.crossmodel import FROZEN_PAIRS
        res = run_crossmodel(Path(args.pairs or FROZEN_PAIRS), Path(args.out), h2=args.h2, n_boot=args.boot,
                             primary_ci=args.primary_ci, main_only=not args.allow_nonmain)
        # no numbers on screen: S is the blinded answer key, so read the files on purpose
        print(f"crossmodel: {Path(args.out) / 'crossmodel.md'}, {Path(args.out) / 'crossmodel.json'}")
        pc = res["primary_ci"]
        print(f"  primary CI: {pc['field']}" + (f" (OVERRIDDEN from the registered {CI_FIELDS[pc['registered']]})"
                                                 if pc["overridden"] else " (the registered one)"))
        for m, v in res["models"].items():
            print(f"  {m}: {v['status']}" + (f" ({'; '.join(v['problems'])})" if v["problems"] else ""))
        for p in res["problems"]:
            print(f"  problem: {p}")
    elif args.cmd == "robots":
        rep = asyncio.run(robot_check(Path(args.out)))
        bad = [c for c in rep["cases"] if not c["pass"]]
        print(f"robot check: {'PASS' if rep['pass'] else 'FAIL'}; {len(rep['cases'])} cases, {len(bad)} failing")
        for c in bad[:20]:
            print("  FAIL", c)


if __name__ == "__main__":
    main()
