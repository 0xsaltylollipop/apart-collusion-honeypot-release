#!/usr/bin/env python3
"""Launch the main honeypot runs exactly as frozen in configs/main-honeypot.json (PREREG Section A).

    .venv/bin/python scripts/launch_main.py check                 # validate the config, print every command, launch nothing
    .venv/bin/python scripts/launch_main.py api claude-haiku-5-5  # an API model: background run with resume loop
    .venv/bin/python scripts/launch_main.py pod oss               # an open model: pod via launch_retry.sh (+ fetch loop)
    ... --dry-run                                                 # print the command only

Every parameter is passed explicitly from the config; a missing key stops the launcher (no CLI or pod default is ever
used: Codex 24 Oct found that defaults give 2 games, seed 1, no k=5 subset and 4096 tokens). Runs are blind (--blind):
the logs show progress and status only. Watch them with scripts/progress.py. Sonnet starts only after Haiku's run is
complete (PREREG A "Order"), unless --force-order.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
CONFIG = CODE / "configs" / "main-honeypot.json"
SENS = CODE / "configs" / "sensitivity.json"
RUN_KEYS = ("games", "seed0", "ks", "primary_k", "sub_ks", "sub_games", "copy_len", "hold", "reps", "steps",
            "step_frac", "c_normal", "c_model", "c_effort", "c_check", "c_script_lead", "debrief", "tag", "blind")
API_KEYS = ("name", "kind", "out", "concurrency", "max_usd", "selfplay")
POD_KEYS = ("name", "kind", "served", "out", "concurrency", "budget_usd", "wall_hours", "c_max_usd",
            "chat_template_kwargs", "selfplay")
TOP_KEYS = ("main_root", "run", "positive_control", "api_sampling", "pod_sampling", "models")


def load(path: Path = CONFIG) -> dict:
    cfg = json.loads(path.read_text())
    missing = [k for k in TOP_KEYS if k not in cfg] + [f"run.{k}" for k in RUN_KEYS if k not in cfg["run"]]
    for m in cfg["models"]:
        need = API_KEYS if m.get("kind") == "api" else POD_KEYS if m.get("kind") == "pod" else ("kind",)
        missing += [f"{m.get('name')}.{k}" for k in need if k not in m]
    for k in ("pc_persona", "pc_steps", "models"):
        if k not in cfg["positive_control"]:
            missing.append(f"positive_control.{k}")
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}; refusing to launch with defaults")
    names = [m["name"] for m in cfg["models"]]
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: duplicate model names {names}")
    return cfg


def model(cfg: dict, name: str) -> dict:
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    raise SystemExit(f"unknown model {name!r} (config has {[m['name'] for m in cfg['models']]})")


def pc_of(cfg: dict, m: dict) -> dict:
    pc = cfg["positive_control"]
    return {"pc_persona": pc["pc_persona"], "pc_steps": pc["pc_steps"]} if m["name"] in pc["models"] else {}


def subject_id(m: dict) -> str:
    return m["name"] if m["kind"] == "api" else m["served"]


def sampling_of(cfg: dict, m: dict) -> dict:
    return dict(cfg["api_sampling"]) if m["kind"] == "api" else pod_sampling(cfg, m)


def frozen_spec(cfg: dict, m: dict, r: dict | None = None, trunks_from: str | None = None) -> dict:
    """The start gate's expectation (honeypot.start_gate): every run parameter, the subject, its sampling and the
    positive control, as the run will see them after its defaults are resolved."""
    r = r or cfg["run"]
    pc = pc_of(cfg, m) if r is cfg["run"] else {}
    return {"model": subject_id(m), "games": r["games"], "seed0": r["seed0"], "ks": sorted(r["ks"]),
            "primary_k": r["primary_k"], "sub_ks": sorted(r["sub_ks"]), "sub_games": r["sub_games"],
            "copy_len": r["copy_len"], "hold": r["hold"], "reps": r["reps"], "steps": r["steps"],
            "step_frac": r["step_frac"], "c_normal": r["c_normal"], "c_model": cfg["run"]["c_model"],
            "c_effort": cfg["run"]["c_effort"], "c_check": cfg["run"]["c_check"],
            "c_script_lead": cfg["run"]["c_script_lead"], "debrief": r["debrief"], "tag": cfg["run"]["tag"],
            "pc_persona": pc.get("pc_persona"), "pc_steps": pc.get("pc_steps", []), "sampling": sampling_of(cfg, m),
            "after_hold": "follow", "private": False, "persona": None,
            "trunks_from": trunks_from,  # main runs: None (they play their own trunks; Astra 24 Oct)
            "blind": r["blind"]}  # checked by the CLI (collusion/cli.py), not by run()


def api_args(cfg: dict, m: dict, r: dict | None = None, out: Path | None = None, max_usd: float | None = None,
             trunks_from: Path | None = None) -> list[str]:
    """`python -m collusion run` for an API model, every parameter explicit (r: a sensitivity row instead of the
    main run)."""
    main = r is None
    r = cfg["run"] if main else {**cfg["run"], **r}
    out = out or Path(cfg["main_root"]) / "runs" / m["out"]
    args = ["run", "--model", m["name"], "--out", str(out), "--games", str(r["games"]), "--seed0", str(r["seed0"]),
            "--ks", ",".join(map(str, r["ks"])), "--primary-k", str(r["primary_k"]),
            "--sub-ks", ",".join(map(str, r["sub_ks"])), "--sub-games", str(r["sub_games"]),
            "--copy-len", str(r["copy_len"]), "--hold", str(r["hold"]), "--reps", str(r["reps"]),
            "--steps", ",".join(r["steps"]), "--step-frac", str(r["step_frac"]), "--c-normal", r["c_normal"],
            "--c-model", r["c_model"], "--c-effort", r["c_effort"], "--c-check", r["c_check"],
            "--c-script-lead", str(r["c_script_lead"]), "--debrief" if r["debrief"] else "--no-debrief",
            "--tag", r["tag"], "--concurrency", str(m["concurrency"]),
            "--max-usd", str(m["max_usd"] if max_usd is None else max_usd),
            "--sampling", json.dumps(cfg["api_sampling"])]
    pc = pc_of(cfg, m) if main else {}
    if pc:
        args += ["--pc-persona", pc["pc_persona"], "--pc-steps", ",".join(pc["pc_steps"])]
    if r["blind"]:
        args.append("--blind")
    if trunks_from is not None:
        args += ["--trunks-from", str(trunks_from)]
    args += ["--frozen-json", json.dumps(frozen_spec(cfg, m, None if main else r,
                                                     None if trunks_from is None else str(trunks_from)))]
    return args


def pod_trap(cfg: dict, m: dict) -> dict:
    """The pod's run JSON (scripts/pod/runner.py trap_args): every key explicit."""
    r = cfg["run"]
    t = {k: r[k] for k in RUN_KEYS}
    t.update(max_usd=m["c_max_usd"], **pc_of(cfg, m), frozen=frozen_spec(cfg, m))
    return t


def sens_rows(cfg: dict, m: dict, scfg: dict | None = None) -> list[dict]:
    """The sensitivity rows for one model: full run specs (common + row), each with its own start-gate spec."""
    scfg = scfg or json.loads(SENS.read_text())
    mine = scfg["models"].get(m["name"])
    if not mine:
        raise SystemExit(f"{m['name']} has no sensitivity rows ({sorted(scfg['models'])})")
    rows = {row["name"]: row for row in scfg["rows"]}
    out = []
    for name in mine["rows"]:
        row = {**scfg["common"], **rows[name]}
        full = {**cfg["run"], **{k: v for k, v in row.items() if k not in ("name", "fork")}}
        src = str(Path(cfg["main_root"]) / "runs" / m["out"]) if row["fork"] else None
        out.append({**row, "frozen": frozen_spec(cfg, m, full, src)})
    return out


def pod_sens(cfg: dict, m: dict) -> list[dict]:
    """The rows as the pod runner reads them (runner.sensitivity_jobs): every run key explicit."""
    scfg = json.loads(SENS.read_text())
    per = scfg["models"][m["name"]]
    rows = []
    for row in sens_rows(cfg, m, scfg):
        t = {k: cfg["run"][k] for k in RUN_KEYS}
        # the pod env is size-limited (RunPod HTTP 500 above ~100 KB): the rows carry every run key explicitly but not
        # their start-gate spec (~0.8 KB each); the main run on the same pod is gated
        t.update({k: v for k, v in row.items() if k != "frozen"})
        t["max_usd"] = per["c_max_usd_per_row"]
        rows.append(t)
    return rows


MODELS_SH = CODE / "scripts" / "pod" / "models.sh"


def models_sh(name: str) -> dict:
    """The serving settings models.sh passes for one model (served id, chat template kwargs, common flags)."""
    text = MODELS_SH.read_text()
    block = re.search(rf"^\s*{re.escape(name)}\) (.*?);;", text, re.S | re.M)
    if not block:
        raise SystemExit(f"models.sh has no entry {name!r}")
    b = block.group(1)
    served = re.search(r"--served (\S+)", b).group(1)
    ctk = re.search(r"--chat-template-kwargs '([^']*)'", b)
    return {"served": served, "chat_template_kwargs": json.loads(ctk.group(1)) if ctk else None,
            "max_tokens": int(re.search(r"--max-tokens (\d+)", text).group(1)),
            "context_cap": int(re.search(r"--max-model-len (\d+)", text).group(1))}


def pod_sampling(cfg: dict, m: dict) -> dict:
    """What the pod's runner will send (runner.trap_args), from models.sh: must equal the self-play manifest."""
    s = models_sh(m["name"])
    samp = {"max_tokens": s["max_tokens"], "context_cap": s["context_cap"], "temperature": 1.0, "top_p": 1.0}
    if s["chat_template_kwargs"] is not None:
        samp["chat_template_kwargs"] = s["chat_template_kwargs"]
    return samp


def check_pod(cfg: dict, m: dict):
    s = models_sh(m["name"])
    want = {**cfg["pod_sampling"], **({"chat_template_kwargs": m["chat_template_kwargs"]}
                                      if m["chat_template_kwargs"] is not None else {})}
    if s["served"] != m["served"] or pod_sampling(cfg, m) != want:
        raise SystemExit(f"{m['name']}: models.sh ({s}) differs from the config ({m['served']}, {want})")


def selfplay_sampling(cfg: dict, m: dict) -> dict:
    """The self-play manifest's sampling (settings only; no outcome is read). A missing folder stops the launcher:
    a honeypot run that cannot be paired is useless (fresh review 24 Oct)."""
    p = Path(cfg["main_root"]) / "runs" / m["selfplay"] / "selfplay" / "f3-chat" / "manifest.json"
    if not p.exists():
        raise SystemExit(f"{m['name']}: self-play manifest {p} is missing; refusing to launch an unpairable run")
    return json.loads(p.read_text())["sampling"]


def require_clean_tree():
    """Launch only from a committed state (fresh review 24 Oct): the code, scripts and configs a run uses must equal
    a commit, which every run records (manifest "git"; pods: the commit at launch)."""
    r = subprocess.run(["git", "status", "--porcelain", "collusion", "scripts", "configs"], cwd=CODE,
                       capture_output=True, text=True)
    if r.returncode != 0:  # fail closed (Codex 24 Oct)
        raise SystemExit(f"git status failed ({r.returncode}): {r.stderr.strip()}; refusing to launch")
    dirty = r.stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted changes in collusion/, scripts/ or configs/; commit first:\n{dirty}")


def run_status(cfg: dict, name: str) -> str | None:
    p = Path(cfg["main_root"]) / "runs" / model(cfg, name)["out"] / "manifest.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def refuse_if_running(cfg: dict, out: Path):
    sys.path.insert(0, str(CODE))
    from collusion.experiments.honeypot import lock_holder  # noqa: PLC0415
    holder = lock_holder(out)
    if holder is not None:
        raise SystemExit(f"{out} is already being run ({holder}); not launching a second writer")


def launch_api(cfg: dict, m: dict, dry: bool, force_order: bool, max_usd: float | None = None):
    if m.get("after") and run_status(cfg, m["after"]) != "complete" and not force_order:
        raise SystemExit(f"{m['name']} starts after {m['after']}, whose run is not complete "
                         f"({run_status(cfg, m['after'])}); check the Anthropic credit, then rerun (or --force-order)")
    sp = selfplay_sampling(cfg, m)
    if sp != cfg["api_sampling"]:
        raise SystemExit(f"{m['name']}: sampling {cfg['api_sampling']} differs from self-play's {sp}")
    refuse_if_running(cfg, Path(cfg["main_root"]) / "runs" / m["out"])
    args = api_args(cfg, m, max_usd=max_usd)  # max_usd: a raised spend stop to resume after a trip (RUNBOOK)
    log = Path(cfg["main_root"]) / "runs" / f"{m['out']}.log"
    cmd = " ".join(shlex.quote(x) for x in [str(CODE / ".venv" / "bin" / "python"), "-m", "collusion", *args])
    # resume loop: the CLI exits nonzero while anything planned is missing (run status "incomplete")
    loop = f"for try in 1 2 3 4 5; do {cmd} && break; sleep 60; done"
    full = f"nohup caffeinate -i bash -c {shlex.quote(loop)} >> {shlex.quote(str(log))} 2>&1 &"
    print(full)
    if not dry:
        subprocess.run(full, shell=True, cwd=CODE, check=True)
        print(f"started {m['name']} -> {log} (watch: scripts/progress.py --all)")


def launch_pod(cfg: dict, m: dict, dry: bool, with_sens: bool = False):
    check_pod(cfg, m)
    sp = selfplay_sampling(cfg, m)
    if sp != pod_sampling(cfg, m):
        raise SystemExit(f"{m['name']}: pod sampling {pod_sampling(cfg, m)} differs from self-play's {sp}")
    trap, wall, budget = pod_trap(cfg, m), m["wall_hours"], m["budget_usd"]
    if with_sens:  # the rows run on this pod after the complete main run (they fork from its trunks)
        per = json.loads(SENS.read_text())["models"][m["name"]]
        trap["sensitivity"] = pod_sens(cfg, m)
        wall, budget = wall + per["extra_wall_hours"], budget + per["extra_budget_usd"]
    env = {**os.environ, "HP_MAIN_ROOT": cfg["main_root"], "WALL": str(wall), "BUDGET": str(budget),
           "CONC": str(m["concurrency"])}
    if dry:
        env["DRY"] = "1"
    cmd = ["bash", str(CODE / "scripts" / "pod" / "launch_retry.sh"), m["name"], m["out"], json.dumps(trap)]
    print(f"HP_MAIN_ROOT={cfg['main_root']} WALL={wall} BUDGET={budget} CONC={m['concurrency']} "
          + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=CODE, env=env, check=False)


def launch_sens_api(cfg: dict, m: dict, dry: bool):
    """An API model's sensitivity rows, one after the other in one background loop (each resumed while incomplete).
    The forked rows need the main run's trunks of their seeds to be finished (checked by the run itself)."""
    per = json.loads(SENS.read_text())["models"][m["name"]]
    main_out = Path(cfg["main_root"]) / "runs" / m["out"]
    py = str(CODE / ".venv" / "bin" / "python")
    loops = []
    for row in sens_rows(cfg, m):
        out = Path(cfg["main_root"]) / "runs" / f"hp-sens-{m['name']}-{row['name']}"
        r = {k: v for k, v in row.items() if k not in ("name", "fork", "frozen")}
        args = api_args(cfg, m, r, out, per["max_usd_per_row"], main_out if row["fork"] else None)
        cmd = " ".join(shlex.quote(x) for x in [py, "-m", "collusion", *args])
        loops.append(f"for try in 1 2 3 4 5; do {cmd} && break; sleep 60; done")
    log = Path(cfg["main_root"]) / "runs" / f"hp-sens-{m['name']}.log"
    full = f"nohup caffeinate -i bash -c {shlex.quote('; '.join(loops))} >> {shlex.quote(str(log))} 2>&1 &"
    print(full)
    if not dry:
        subprocess.run(full, shell=True, cwd=CODE, check=True)
        print(f"started the sensitivity rows of {m['name']} -> {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "api", "pod", "sens"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-order", action="store_true")
    ap.add_argument("--with-sens", action="store_true", help="pod: run the sensitivity rows after the main run")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="api: resume after a spend-stop trip with this higher stop (RUNBOOK section 4)")
    ap.add_argument("--config", default=str(CONFIG))
    a = ap.parse_args()
    cfg = load(Path(a.config))
    if a.cmd == "check":
        for m in cfg["models"]:
            if m["kind"] == "api":
                sp = selfplay_sampling(cfg, m)
                ok = "matches self-play" if sp == cfg["api_sampling"] else f"DIFFERS from self-play {sp}"
                print(f"[api] {m['name']} ({ok}): python -m collusion " + " ".join(shlex.quote(x) for x in api_args(cfg, m)))
            else:
                check_pod(cfg, m)
                sp = selfplay_sampling(cfg, m)
                ok = "matches self-play" if sp == pod_sampling(cfg, m) else f"DIFFERS from self-play {sp}"
                print(f"[pod] {m['name']} ({ok}): WALL={m['wall_hours']} BUDGET={m['budget_usd']} "
                      f"CONC={m['concurrency']} trap={json.dumps(pod_trap(cfg, m))}")
        return
    if not a.name:
        ap.error("give a model name")
    m = model(cfg, a.name)
    if not a.dry_run:
        require_clean_tree()
    if a.cmd == "sens":
        if m["kind"] != "api":
            raise SystemExit(f"{a.name} is a pod model: its rows run on its main pod (`pod {a.name} --with-sens`)")
        return launch_sens_api(cfg, m, a.dry_run)
    if m["kind"] != a.cmd:
        raise SystemExit(f"{a.name} is a {m['kind']} model; use `{m['kind']}`")
    (launch_api(cfg, m, a.dry_run, a.force_order, a.max_usd) if a.cmd == "api"
     else launch_pod(cfg, m, a.dry_run, a.with_sens))


if __name__ == "__main__":
    sys.exit(main())
