#!/usr/bin/env python3
"""The laptop side of the Modal fallback (modal_app.py): the same commands as pod.py, for when RunPod has no hosts.

    python scripts/pod/modal_pod.py --out-name <name> launch <the pod.py launch flags> [--gpu H100] [--dry-run]
    python scripts/pod/modal_pod.py --out-name <name> fetch [--loop S]   # pull runs/ from the Volume (progress only)
    python scripts/pod/modal_pod.py --out-name <name> status
    python scripts/pod/modal_pod.py --out-name <name> down               # stop this run's Modal App
    BACKEND=modal scripts/pod/models.sh <model> <out-name> '<run JSON>'  # the models' exact settings, on Modal

The launch flags are pod.py's own (pod.add_launch_args) and the run config is pod.run_cfg's, so a model runs with the
same weights, revision, parser, chat template kwargs and sampling as on a pod; only "platform", "cost_per_hr" and the
GPU differ. Needs the modal package in this venv and a Modal login (`modal token new`). Results land in runs/<name>/
in the layout pod.py fetch produces. Keys are read from .env inside modal_app.py and never printed.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pod  # noqa: E402

# Modal list prices, 24 Oct (modal.com/pricing, per second x 3600), plus the function's 8 CPU cores and 32 GiB.
GPU_USD_H = {"L4": 0.80, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "RTX-PRO-6000": 3.03, "H100": 3.95,
             "H200": 4.54}
CPU_MEM_USD_H = 8 * 0.0000131 * 3600 + 32 * 0.00000222 * 3600  # ~$0.63/h
DEFAULT_GPU = "L40S"
VLLM_LOAD_H = 0.75  # vLLM start-up inside the wall clock's margin (the pod's backstop uses the same)


def meta_dir() -> Path:
    return pod.OUT / "_modal"


def rec_path() -> Path:
    return pod.OUT / "modal.json"


def load():
    return json.loads(rec_path().read_text()) if rec_path().exists() else None


def save(r):
    pod.OUT.mkdir(parents=True, exist_ok=True)
    tmp = rec_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=2))
    tmp.replace(rec_path())


def rate(gpu: str) -> float:
    if gpu not in GPU_USD_H:
        raise SystemExit(f"unknown Modal GPU {gpu!r} (known: {', '.join(GPU_USD_H)})")
    return round(GPU_USD_H[gpu] + CPU_MEM_USD_H, 3)


def modal_cfg(a) -> dict:
    """pod.run_cfg plus the Modal-only keys: the runner's wall clock uses cost_per_hr (no RunPod API to ask)."""
    cfg = {**pod.run_cfg(a), "git_commit": pod.git_commit(), "platform": "modal", "cost_per_hr": rate(a.gpu)}
    # The pod budgets follow RUNBOOK's rule BUDGET >= WALL x $2.70/h, so min(WALL, BUDGET / rate) never cuts a run
    # short on a RunPod GPU. Modal's H100 costs ~$4.6/h: keep the same rule at Modal's rate (WALL stays the breaker).
    floor = round(cfg["wall_hours"] * cfg["cost_per_hr"], 2)
    if cfg["budget_usd"] < floor:
        cfg["budget_usd_pod"], cfg["budget_usd"] = cfg["budget_usd"], floor
    if cfg["wall_hours"] + VLLM_LOAD_H > 24:
        raise SystemExit(f"wall_hours {cfg['wall_hours']} + vLLM start-up exceeds Modal's 24 h function limit")
    return cfg


def key_names(cfg: dict) -> list[str]:
    """The OpenAI keys go along only when the subject or firm C is an OpenAI model (as pod.pod_env)."""
    llm = pod.llm_module()
    if any(m and llm.route(m) == "openai" for m in (cfg.get("served"), (cfg.get("trap") or {}).get("c_model"))):
        return list(llm.KEYS["openai"])
    return []


def modal_cmd(*args) -> list[str]:
    return [sys.executable, "-m", "modal", *args]


def launch_cmd(cfg: dict, gpu: str, cfg_file: Path) -> tuple[list[str], dict]:
    timeout_h = cfg["wall_hours"] + VLLM_LOAD_H + 0.25
    cmd = modal_cmd("run", "--detach", str(HERE / "modal_app.py"), "--out-name", pod.OUT.name, "--cfg-file",
                    str(cfg_file), "--gpu", gpu, "--timeout-h", str(timeout_h), "--keys", ",".join(key_names(cfg)))
    env = {"HP_MODAL_APP": f"collusion-{pod.OUT.name}"[:60], "HP_MAIN_ROOT": str(pod.MAIN)}
    return cmd, env


def launch(a):
    if a.dry_run:  # writes nothing (no run folder, no lock file before a real launch)
        cfg = modal_cfg(a)
        cmd, _ = launch_cmd(cfg, a.gpu, meta_dir() / "cfg.json")
        print(json.dumps(cfg, indent=2))
        print(" ".join(cmd))
        print(pod.prereg_text(cfg)[:1500])
        return
    with pod.launch_lock():  # shared with pod.py: check -> launch -> record is atomic per out-name (Astra 24 Oct)
        _launch(a)


def _launch(a):
    live = pod.live_record()  # this out-name launched on either backend and not ended
    if live:
        print(f"{live[0]} records a launched run that has not ended ({live[1].get('app_id') or live[1].get('id')}); "
              "not launching another")
        return
    cfg = modal_cfg(a)
    md = meta_dir()
    cfg_file = md / "cfg.json"
    cmd, env = launch_cmd(cfg, a.gpu, cfg_file)
    md.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps(cfg))
    (md / "app.json").unlink(missing_ok=True)
    if not (pod.OUT / "RESULTS.md").exists():  # the pre-registered plan goes in before any data arrives
        (pod.OUT / "RESULTS.md").write_text(pod.prereg_text(cfg) + f"\nCode commit at launch: {cfg['git_commit']}\n"
                                            "Backend: Modal\n\n## Results\n\n(to be filled after the run)\n")
    log = open(md / "launch.log", "a")
    proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)  # --detach: the App keeps running if this client goes away
    t = time.time()
    rec = {"app_id": None, "gpu": a.gpu, "created": t, "cost_per_hr": cfg["cost_per_hr"],
           "budget_usd": cfg["budget_usd"], "wall_hours": cfg["wall_hours"], "client_pid": proc.pid,
           "git_commit": cfg["git_commit"], "served": cfg["served"], "c_model": cfg["trap"].get("c_model"),
           "phase": "launching"}
    save(rec)  # recorded before the wait (Codex 24 Oct): a launcher interrupted while the App starts leaves a live
    #            record, so a retry is refused instead of starting a second App on the same Volume folder
    while time.time() - t < 20 * 60 and not (md / "app.json").exists() and proc.poll() is None:
        time.sleep(5)
    app = json.loads((md / "app.json").read_text()) if (md / "app.json").exists() else {}
    save({**rec, "app_id": app.get("app_id"), "phase": "launched" if app else "launch unconfirmed"})
    if app:
        print(f"Modal app {app['app_id']} on {a.gpu} ({cfg['cost_per_hr']} $/h, budget ${cfg['budget_usd']})")
    else:
        print(f"no App id after {time.time() - t:.0f}s (client exit {proc.poll()}); see {md / 'launch.log'}; "
              "the fetch loop picks the id up if the client is still starting the App")


QUEUE_LIMIT_H = 2.0  # no container started this long after launch (no GPU free, image build stuck): stop the App


def run_start(r):
    """When the container started (the runner's t0 on the Volume): spend and the wall clock count from here, not
    from the launch, so time queued for a GPU never shortens the run (Codex 24 Oct)."""
    return r.get("t0")


def accrued(r, now=None):
    t0 = run_start(r)
    if not t0:
        return 0.0
    end = r.get("ended") or now or time.time()
    return (r.get("cost_per_hr") or 0) * max(0.0, end - t0) / 3600


def stop_app(r) -> bool:
    reconcile(r)
    if not r.get("app_id"):
        print("no App id recorded; stop it in the Modal web UI (app name collusion-<out-name>)")
        return False
    res = subprocess.run(modal_cmd("app", "stop", r["app_id"], "-y"), capture_output=True, text=True)
    print(f"modal app stop {r['app_id']}: exit {res.returncode}")
    return res.returncode == 0


def reconcile(r):
    """Pick up an App id written after launch() stopped waiting (a slow first image build)."""
    app_f = meta_dir() / "app.json"
    if not r.get("app_id") and app_f.exists():
        r["app_id"] = json.loads(app_f.read_text()).get("app_id")


def fetch_once():
    r = load()
    if not r:
        print("no Modal run recorded")
        return None
    if r.get("ended"):
        print(f"Modal run {r.get('app_id')} ended ({r.get('phase')}); cost ~${accrued(r):.2f}")
        return r
    now = time.time()
    reconcile(r)
    synced = False
    try:
        with tempfile.TemporaryDirectory() as td:
            res = subprocess.run(modal_cmd("volume", "get", "--force", "collusion-runs", f"{pod.OUT.name}/runs", td),
                                 capture_output=True, text=True, timeout=1800)
            if res.returncode == 0 and (Path(td) / "runs").exists():
                pod.place(Path(td) / "runs")
                r["last_fetch"], synced = now, True
                r.pop("last_error", None)
            else:
                r["last_error"] = f"volume get exit {res.returncode}: {res.stderr[-200:]}"
    except Exception as e:
        r["last_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    st_f = pod.OUT / "_pod" / "state.json"
    if st_f.exists():
        st = json.loads(st_f.read_text())
        r["phase"], r["progress"] = st.get("phase"), st.get("progress")
        r["t0"] = st.get("t0") or r.get("t0")
    client_done = bool(r.get("client_pid")) and not _alive(r["client_pid"])
    final = r.get("phase") in ("done", "timeout", "error") or r.get("stop_confirmed")
    if client_done and synced and final and r.get("final_seen"):
        # the function had returned (or was stopped) before this fetch began, so this full download holds its last
        # Volume commit (Codex 24 Oct: a download that spans the end could mix earlier files with the final state)
        r["ended"] = now
    elif client_done and synced and final:
        r["final_seen"] = now  # one more full download on the next fetch before the record ends
    elif client_done and not r.get("app_id") and not run_start(r):
        r["ended"], r["phase"] = now, "launch failed (no App id, client gone)"
    elif client_done and r.get("stop_confirmed") and not run_start(r) and not synced and _remote_output_absent():
        # stopped while still queued: no container ever ran, so the Volume has no folder to download (Codex 24 Oct:
        # the record never ended and blocked both backends for this out-name)
        r["ended"], r["phase"] = now, "stopped before any container started (no output)"
    elif not r.get("stop_confirmed"):
        t0 = run_start(r)
        if t0 and accrued(r, now) > r["budget_usd"] * 1.05:
            why = "budget"  # laptop-side breaker, as pod.py's finalize; later fetches pull the rest
        elif t0 and now - t0 > (r["wall_hours"] + VLLM_LOAD_H + 1.0) * 3600:
            why = "wall clock"
        elif not t0 and now - r["created"] > QUEUE_LIMIT_H * 3600:
            why = "no container after the queue limit"
        else:
            why = None
        if why:
            r["stop_reason"], r["stop_confirmed"] = why, stop_app(r)
    save(r)
    print(f"{time.strftime('%H:%M:%S')} modal {r.get('app_id')} phase={r.get('phase')} progress={r.get('progress')} "
          f"cost ~${accrued(r):.2f}/{r['budget_usd']} err={r.get('last_error', '')}")
    return r


def _remote_output_absent() -> bool:
    """True only on affirmative evidence that this run never wrote to the Volume: its root lists no <out-name>/, or
    <out-name>/ has no runs/. A failed listing is never evidence (Codex 24 Oct: a download error after a stop must not
    end the record while results may sit on the Volume)."""
    def ls(path):
        try:
            res = subprocess.run(modal_cmd("volume", "ls", "collusion-runs", path), capture_output=True, text=True,
                                 timeout=300)
        except Exception:
            return None
        if res.returncode != 0:
            return None
        return {Path(line.strip()).name for line in res.stdout.splitlines() if line.strip()}  # one path per line
    root = ls("/")
    if root is None:
        return False
    if pod.OUT.name not in root:
        return True
    sub = ls(pod.OUT.name)
    return sub is not None and "runs" not in sub


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def fetch(a):
    while True:
        try:
            r = fetch_once()
        except Exception as e:
            if not a.loop:
                raise
            print(f"{time.strftime('%H:%M:%S')} fetch failed ({type(e).__name__}: {str(e)[:120]}); retrying", flush=True)
            time.sleep(a.loop)
            continue
        if not a.loop or not r or r.get("ended"):
            return
        time.sleep(a.loop)


def status(_):
    r = load()
    if not r:
        print("no Modal run recorded")
        return
    reconcile(r)
    print(json.dumps({k: r.get(k) for k in ("app_id", "gpu", "phase", "progress", "cost_per_hr", "budget_usd",
                                            "stop_reason", "ended")}))
    print(f"cost so far ~${accrued(r):.2f} (list prices x time since the container started)")


def down(_):
    r = load()
    if not r:
        print("no Modal run recorded")
        return
    if stop_app(r):  # the record ends only after a fetch has pulled the final Volume state (fetch_once)
        r["stop_confirmed"], r["stop_reason"] = True, "down"
        print("stopped; run `fetch` until it reports the run ended (the last results are on the Volume)")
    else:
        print("stop NOT confirmed: the App may still run (check the Modal web UI); monitoring continues")
    save(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", required=True, help="results go to runs/<out-name>/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch")
    pod.add_launch_args(la)
    la.add_argument("--gpu", default=DEFAULT_GPU, help=f"Modal GPU type, one of {', '.join(GPU_USD_H)}")
    f = sub.add_parser("fetch")
    f.add_argument("--loop", type=int, default=0)
    sub.add_parser("status")
    sub.add_parser("down")
    a = ap.parse_args()
    pod.set_out(a.out_name)
    {"launch": launch, "fetch": fetch, "status": status, "down": down}[a.cmd](a)


if __name__ == "__main__":
    main()
