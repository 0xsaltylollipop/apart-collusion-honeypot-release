#!/usr/bin/env python3
"""Runs ON a RunPod GPU pod: vLLM + the honeypot run and/or self-play (python -m collusion run | selfplay), a
token-gated results server, and self-termination (same lifecycle as the pre-sprint pod runner):

  starting -> serving -> playing -> done | timeout | error -> wait for POST /ack (laptop fetched the final
  tarball) -> DELETE own pod. No ack within ACK_WAIT -> DELETE anyway (the laptop fetches results periodically;
  hp.py also deletes from the laptop side). A bash backstop deletes the pod if this process hangs.

The honeypot run is resumable (finished trunks/copies are skipped), so a crashed run process is simply restarted (at
most 3 times); every self-play scenario likewise (at most 5 times).

Endpoints on :8001 (Authorization: Bearer $RESULTS_TOKEN): GET /status, GET /results.tgz, POST /ack,
POST /finalize (stop now, e.g. budget reached).
"""

import hmac
import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

W = Path(os.environ.get("POD_WORKSPACE", "/workspace"))
RUNS = W / "runs"
OUT = RUNS / "honeypot"
HARNESS = Path(os.environ.get("HARNESS_DIR") or W / "boot" / "code")  # the collusion package, shipped by pod.py
STATE_F = RUNS / "state.json"
CFG = json.loads(os.environ["HP_CFG"])
TOKEN = os.environ["RESULTS_TOKEN"]
POD = os.environ.get("RUNPOD_POD_ID")
# 24 Oct: Modal as the fallback GPU source (scripts/pod/modal_app.py). There the hourly rate comes from the launch
# config, the wall clock counts from the first start (Modal restarts a preempted input), and there is no laptop ack
# or self-termination: the function returns and its Volume keeps the results.
ON_MODAL = CFG.get("platform") == "modal"
REST = "https://rest.runpod.io/v1"
T0 = time.time()
LOCK = threading.Lock()
STOP = threading.Event()
WALL_H = None
VLLM = None

state = {"phase": "starting", "model": CFG["model"], "served": CFG["served"], "pod": POD,
         "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0)), "acked": False, "log": []}


def log(msg):
    line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
    print(line, flush=True)
    with LOCK:
        state["log"] = (state["log"] + [line])[-60:]
    save()


def save():
    RUNS.mkdir(parents=True, exist_ok=True)
    with LOCK:
        tmp = STATE_F.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(STATE_F)


def set_phase(p):
    with LOCK:
        state["phase"] = p
    log(f"phase={p}")


def rp(method, path, body=None):
    last = None
    for name in ("RUNPOD_API_KEY", "SELF_KEY"):
        key = os.environ.get(name)
        if not key:
            continue
        req = urllib.request.Request(f"{REST}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("User-Agent", "apart-collusion-sprint/0.1")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return name, (json.loads(raw) if raw else None)
        except Exception as e:
            last = f"{name}: {type(e).__name__} {getattr(e, 'code', '')}"
    raise RuntimeError(f"RunPod API {method} {path} failed ({last})")


def own_cost_per_hr():
    try:
        _, pod = rp("GET", f"/pods/{POD}")
        return float(pod.get("costPerHr") or 0) or None
    except Exception as e:
        log(f"cannot read own pod info: {e}")
        return None


def self_end(acked):
    """Always TERMINATE (never stop): the laptop fetches results periodically, and a stopped pod that is later
    restarted would rerun the pilot with a fresh clock."""
    if not POD:
        log("no RUNPOD_POD_ID; cannot self-terminate")
        return
    log("results acknowledged: terminating pod" if acked else "no ack in time: terminating pod anyway")
    for _ in range(3):
        try:
            rp("DELETE", f"/pods/{POD}")
            return
        except Exception as e:
            log(f"self-terminate failed: {e}")
            time.sleep(30)


def progress():
    """Counts from the files on disk; unit-aware (each branch's own fork point from its metadata)."""
    p = {"trunks_done": 0, "branches_done": 0, "branches_started": 0, "posthoc_done": 0}
    if CFG.get("mode") == "trap":  # trunks are played to k only; copies under copies/<unit>/<step>-b<i>
        p["trunks_started"] = len(list((OUT / "trunks").glob("trunk-s*")))
        for cd in (OUT / "copies").glob("*/*"):
            p["branches_started"] += 1
            try:
                meta = json.loads((cd / "metadata.json").read_text())
                summ = json.loads((cd / "summary.json").read_text())
                k = meta["parent"]["at_period"]
                p["branches_done"] += summ.get("error") is None and summ.get("periods") == k + CFG["trap"].get(
                    "copy_len", 30)
            except Exception:
                pass
        sp = 0
        for f in (OUT / "selfplay").rglob("sp-s*/rounds.jsonl"):
            try:
                sp += sum(1 for _ in open(f)) - 1
            except Exception:
                pass
        p["selfplay_rounds"] = sp
        return p
    for s in (OUT / "trunks").glob("trunk-s*/summary.json"):
        try:
            p["trunks_done"] += json.loads(s.read_text()).get("error") is None
        except Exception:
            pass
    for bd in (OUT / "branches").glob("*/*"):
        p["branches_started"] += 1
        try:
            meta = json.loads((bd / "metadata.json").read_text())
            summ = json.loads((bd / "summary.json").read_text())
            p["branches_done"] += summ.get("error") is None and summ.get("periods") == meta["config"]["periods"]
        except Exception:
            pass
        p["posthoc_done"] += (bd / "posthoc.json").exists()
    return p


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _auth(self):
        if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {TOKEN}"):
            self.send_response(401)
            self.end_headers()
            return False
        return True

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if not self._auth():
            return
        if self.path.startswith("/status"):
            with LOCK:
                s = dict(state)
            s["progress"] = progress()
            s["uptime_h"] = round((time.time() - T0) / 3600, 3)
            self._json(s)
        elif self.path.startswith("/results.tgz"):
            fd, name = tempfile.mkstemp(dir=W, prefix="results-", suffix=".tgz")  # one file per request
            os.close(fd)
            tmp = Path(name)
            try:
                with tarfile.open(tmp, "w:gz") as tf:
                    tf.add(RUNS, arcname="runs")
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(tmp.stat().st_size))
                self.end_headers()
                with open(tmp, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            finally:
                tmp.unlink(missing_ok=True)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not self._auth():
            return
        if self.path.startswith("/ack"):
            with LOCK:
                done = state["phase"] in ("done", "timeout", "error")
                if done:
                    state["acked"] = True
            save()
            self._json({"acked": done})
        elif self.path.startswith("/finalize"):
            log("finalize requested by the laptop")
            STOP.set()
            self._json({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()


def serve():
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", 8001), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def over_wall():
    return WALL_H is not None and time.time() - T0 > WALL_H * 3600


def start_vllm():
    global VLLM
    args = ["vllm", "serve", CFG["model"], "--host", "127.0.0.1", "--port", "8000",
            "--served-model-name", CFG["served"], "--max-model-len", str(CFG["max_model_len"]),
            "--reasoning-parser", CFG["parser"], "--generation-config", "vllm", "--enable-prefix-caching",
            "--gpu-memory-utilization", "0.92", "--revision", CFG["revision"],
            "--tokenizer-revision", CFG["revision"]] + CFG.get("extra_vllm", [])
    logf = open(RUNS / "vllm.log", "a")
    VLLM = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"vllm starting (max_model_len={CFG['max_model_len']})")
    t = time.time()
    while time.time() - t < 45 * 60:
        if VLLM.poll() is not None:
            raise RuntimeError(f"vllm exited with {VLLM.returncode}; see vllm.log")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as r:
                if r.status == 200:
                    log(f"vllm ready after {time.time() - t:.0f}s")
                    return
        except Exception:
            pass
        if STOP.is_set() or over_wall():
            raise TimeoutError("stopped while vllm was loading")
        time.sleep(10)
    raise TimeoutError("vllm did not come up in 45 min")


def stop_vllm():
    global VLLM
    if VLLM and VLLM.poll() is None:
        os.killpg(VLLM.pid, signal.SIGTERM)
        try:
            VLLM.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(VLLM.pid, signal.SIGKILL)
    VLLM = None


def game_env():
    env = dict(os.environ)
    env.update({"VLLM_BASE_URL": "http://127.0.0.1:8000/v1", "VLLM_API_KEY": "EMPTY",
                "VLLM_MODEL_REVISION": f"{CFG['model']}@{CFG.get('revision')}",
                "BERTRAND_GIT_COMMIT": CFG.get("git_commit", ""), "PYTHONPATH": str(HARNESS),
                "PYTHONUNBUFFERED": "1"})
    for k in ("RESULTS_TOKEN", "SELF_KEY", "RUNPOD_API_KEY", "HP_CFG"):
        env.pop(k, None)
    return env


MODULE = "collusion"


def run_cmd(args, name):
    logf = open(RUNS / f"{name}.log", "a")
    return subprocess.Popen([sys.executable, "-m", MODULE, *args], cwd=HARNESS, env=game_env(),
                            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)


def kill(p):
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def trap_args(t=None, out=None):
    """CLI for `python -m collusion run` (the honeypot): CFG["trap"] (or a sensitivity row `t`) holds games, ks,
    copy_len, reps, steps, c_model, c_effort, seed0, tag, private, hold, persona, pc_persona, pc_steps, c_check,
    c_script_lead, primary_k, h1, debrief, sub_ks, sub_games, step_frac, c_normal, max_usd, blind, frozen; vLLM
    sampling as in self-play. Main runs pass "frozen", so a default used here by mistake stops the run at its start
    gate instead of producing data."""
    t = CFG["trap"] if t is None else t
    out = OUT if out is None else out
    sampling = {"max_tokens": CFG["max_tokens"], "context_cap": CFG["max_model_len"], "temperature": 1.0,
                "top_p": 1.0}
    if CFG.get("chat_template_kwargs") is not None:
        sampling["chat_template_kwargs"] = CFG["chat_template_kwargs"]
    args = ["run", "--model", CFG["served"], "--out", str(out), "--concurrency", str(CFG["concurrency"]),
            "--games", str(t.get("games", 2)), "--ks", ",".join(map(str, t.get("ks", [10]))),
            "--copy-len", str(t.get("copy_len", 30)), "--reps", str(t.get("reps", 1)),
            "--steps", ",".join(t.get("steps", ["none", "silent", "hint", "offer", "benign"])),
            "--c-model", t.get("c_model", "gpt-5.4-mini-2026-03-17"), "--c-effort", t.get("c_effort", "low"),
            "--seed0", str(t.get("seed0", 1)), "--tag", t.get("tag", "pre-sprint-trap"),
            "--sampling", json.dumps(sampling)]
    if t.get("private"):
        args.append("--private")
    if t.get("hold") is not None:
        args += ["--hold", str(t["hold"])]
    if t.get("persona"):  # old positive control (e.g. "colluder") from round 1, subjects only
        args += ["--persona", t["persona"]]
    if t.get("pc_persona"):  # positive control from k+1 only (24 Oct), extra copies of pc_steps
        args += ["--pc-persona", t["pc_persona"], "--pc-steps", ",".join(t.get("pc_steps", ["none", "offer"]))]
    if "c_check" in t:  # pre-send checker of C's trunk messages ("" or null = legacy regex guard)
        args += ["--c-check", t["c_check"] or ""]
    for key in ("c_script_lead", "primary_k"):
        if t.get(key) is not None:
            args += [f"--{key.replace('_', '-')}", str(t[key])]
    if t.get("h1"):
        args += ["--h1", t["h1"]]
    if t.get("debrief") is not None:  # suspicion probe after each copy (default on for v3 runs of a model)
        args += ["--debrief" if t["debrief"] else "--no-debrief"]
    if t.get("sub_ks"):  # extra save points in the first sub_games games only (24 Oct: k=5 in 3 of 10 games)
        args += ["--sub-ks", ",".join(map(str, t["sub_ks"])), "--sub-games", str(t["sub_games"])]
    if t.get("step_frac") is not None:  # sensitivity subset: C's raise (1.08 / 1.20)
        args += ["--step-frac", str(t["step_frac"])]
    if t.get("c_normal"):  # sensitivity subset: C's normal rule (best_response)
        args += ["--c-normal", t["c_normal"]]
    if t.get("max_usd") is not None:  # spend stop of the run's API calls (firm C and its checker on a pod)
        args += ["--max-usd", str(t["max_usd"])]
    if t.get("blind"):  # main runs: no unit prices or analysis in the pod's logs
        args.append("--blind")
    if t.get("frozen") is not None:  # main runs: the start gate (configs/main-honeypot.json, PREREG A)
        args += ["--frozen-json", json.dumps(t["frozen"])]
    if t.get("trunks_from"):  # sensitivity rows: fork from the main run's trunks on this pod
        args += ["--trunks-from", t["trunks_from"]]
    return args


def sensitivity_jobs():
    """The sensitivity subset's rows on this pod, run one after the other once the main run is complete
    (CFG["trap"]["sensitivity"], from configs/sensitivity.json via scripts/launch_main.py): each row is a full run spec
    (its own out folder next to the main one, in the results tarball); rows with "fork": true branch from the
    main run's trunks (paired units), the others play their own trunks (their own seeds)."""
    jobs = []
    for row in CFG["trap"].get("sensitivity") or []:
        spec = {k: v for k, v in row.items() if k not in ("name", "fork")}
        if row.get("fork"):
            spec["trunks_from"] = str(OUT)
        # a sibling of the main run's folder (never inside it: its cost and file scans would count the rows); the
        # laptop's fetch puts it under runs/<out-name>/_pod/sens/<row>
        jobs.append((row["name"], trap_args(spec, RUNS / "sens" / row["name"]), RUNS / "sens" / row["name"]))
    return jobs


def selfplay_jobs():
    """CLIs for `python -m collusion selfplay` (answer key; same sampling as the trap subjects), as
    [(name, args)]. CFG["trap"]["selfplay"] is one scenario {"games", "periods", "seed0", "chat", "tag"} (out: selfplay/)
    or a list of scenarios that also carry "name" and "firms" (out: selfplay/<name>/), all run side by side."""
    sps = CFG["trap"].get("selfplay")
    if not sps:
        return []
    sampling = {"max_tokens": CFG["max_tokens"], "context_cap": CFG["max_model_len"], "temperature": 1.0,
                "top_p": 1.0}
    if CFG.get("chat_template_kwargs") is not None:
        sampling["chat_template_kwargs"] = CFG["chat_template_kwargs"]
    if isinstance(sps, list):
        names = [sp.get("name") for sp in sps]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError(f"self-play scenarios need unique, non-empty names: {names}")
    jobs = []
    for sp in (sps if isinstance(sps, list) else [sps]):
        name = sp.get("name") if isinstance(sps, list) else None
        out = OUT / "selfplay" / name if name else OUT / "selfplay"
        args = ["selfplay", "--model", CFG["served"], "--out", str(out),
                "--games", str(sp.get("games", 3)), "--periods", str(sp.get("periods", 300)),
                "--seed0", str(sp.get("seed0", 101)), "--firms", str(sp.get("firms", 3)),
                "--concurrency", str(CFG["concurrency"]), "--sampling", json.dumps(sampling)]
        if sp.get("chat"):
            args.append("--chat")
        tag = sp.get("tag") or CFG["trap"].get("tag")  # a top-level run tag also labels the self-play games
        if tag:
            args += ["--tag", tag]
        jobs.append((f"selfplay-{name}" if name else "selfplay", args))
    return jobs


def play():
    if CFG.get("no_vllm"):  # fake/OpenAI subjects (smoke tests): no local model server
        log("no_vllm: skipping the vLLM server")
    else:
        start_vllm()
    set_phase("playing")
    if CFG.get("mode") != "trap":
        raise ValueError("only the honeypot/self-play mode is supported (the older pilot designs are in the archive repo)")
    trap = True
    pilot_args_ = trap_args()
    jobs = selfplay_jobs()
    procs = {}
    for name, args in jobs:
        procs[name] = run_cmd(args, name)
        log(f"self-play running ({name}): {' '.join(args)}")
    try:
        if trap and CFG["trap"].get("games", 2) == 0:  # self-play only
            status = "done"
        else:
            status = _play_trap_or_pilot(pilot_args_)
    except BaseException:
        for p in procs.values():
            kill(p)
        raise
    if procs and status == "done":
        return _wait_selfplay(dict(jobs), procs)
    for p in procs.values():
        kill(p)
    return status


def _wait_selfplay(args_by_name, procs):
    """The trap run is finished (or absent): wait for every self-play scenario (up to 5 attempts each)."""
    tries = {name: 1 for name in procs}
    while True:
        running = [n for n, p in procs.items() if p.poll() is None]
        for name, p in procs.items():
            if p.poll() is not None and p.returncode != 0 and tries[name] < 5:
                tries[name] += 1
                log(f"{name} exited with {p.returncode}; resuming in 60 s (attempt {tries[name]})")
                time.sleep(60)
                procs[name] = run_cmd(args_by_name[name], name)
                running.append(name)
        if not running:
            break
        time.sleep(CFG.get("poll_s", 30))
        with LOCK:
            state["progress"] = progress()
        save()
        if STOP.is_set() or over_wall():
            log("stopping self-play (finalize or wall-clock limit)")
            for p in procs.values():
                kill(p)
            return "timeout"
        if VLLM is not None and VLLM.poll() is not None:
            for p in procs.values():
                kill(p)
            raise RuntimeError("vllm died during self-play")
    failed = [n for n, p in procs.items() if p.returncode != 0]
    if failed:
        log(f"self-play failed after 5 attempts: {failed}")
        return "error"
    return "done"


def _play_trap_or_pilot(pilot_args_):
    """The honeypot run, resumed while it exits nonzero (the CLI exits 3 while any planned trunk, copy or debrief is
    missing; Codex 24 Oct; see _run_until_complete), then the sensitivity rows. "done" only when the main run and every
    row are complete; otherwise "error" or "timeout" (results are still synced: the laptop fetches the tarball before
    the ack)."""
    st = _run_until_complete(pilot_args_, "pilot", OUT)
    with LOCK:
        state["main_status"] = st  # reported separately from the sensitivity rows (Astra 24 Oct)
    jobs = sensitivity_jobs()
    # main PLAY complete (only debriefs missing; a spend-stop note does not count): the rows can fork
    play_done = st == "done" or set(_missing(OUT)) - {"budget_stop"} == {"debriefs"}
    if st == "timeout" or not play_done or (st != "done" and not jobs):
        return st
    for name, args, out in jobs:  # after the main play, one row at a time
        st = _run_until_complete(args, f"sens-{name}", out)
        log(f"sensitivity row {name}: {st}")
        with LOCK:
            state.setdefault("sensitivity", {})[name] = st
        if st == "timeout":
            return "timeout"
    if state["main_status"] != "done":  # only its debriefs were missing: one more resume after the rows
        st = _run_until_complete(pilot_args_, "pilot", OUT)
        with LOCK:
            state["main_status"] = st
        if st != "done":
            return st
    failed = [n for n, s in (state.get("sensitivity") or {}).items() if s != "done"]
    if failed:  # the pod is never reported "done" with a missing row; main_status says the main run is complete
        log(f"sensitivity rows not complete: {failed}; phase error (main run complete)")
        return "error"
    if CFG["trap"].get("blind"):  # main runs: analysed on the laptop after unblinding, never printed here
        log("blind run: no analysis on the pod")
        return "done"
    a = run_cmd(["analyze", str(OUT)], "analyze")
    a.wait(600)
    log(f"analysis exit {a.returncode}")
    return "done"


def _missing(out) -> dict:
    """What the run's manifest says is still missing ({} when complete or unreadable)."""
    try:
        return json.loads((Path(out) / "manifest.json").read_text()).get("incomplete") or {}
    except Exception:
        return {}


def _work_done(out) -> int:
    """Records the run has written so far: rounds of every trunk and copy, and debrief answers, failed tries and
    give-up markers. It grows with any progress, also inside a debrief whose copy is not yet complete, which the
    missing counts cannot show (Astra 24 Oct)."""
    n = 0
    for fname in ("rounds.jsonl", "debrief.jsonl"):
        for p in Path(out).rglob(fname):
            try:
                with open(p, "rb") as f:
                    n += sum(1 for _ in f)
            except OSError:
                pass
    return n


def _run_until_complete(args, name, out=None):
    """One `python -m collusion run`, resumed while it exits nonzero: as long as the wall clock allows and each
    resume makes progress (fresh review 24 Oct: not a fixed number of tries); it gives up after
    CFG["trap"]["stalls"] (default 2) resumes in a row without progress, or CFG["trap"]["attempts"] (default 40).
    Progress = what is missing shrank, or the run wrote new records (rounds, debrief answers, failed tries, give-up
    markers; the debrief's tries are bounded, so this cannot go on forever; Astra 24 Oct). "done" once it exits 0
    (complete), "timeout" on the wall clock or finalize, else "error"."""
    attempts, stalls_max = int(CFG["trap"].get("attempts", 40)), int(CFG["trap"].get("stalls", 2))
    before, work_before, stalls = None, None, 0
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            time.sleep(CFG.get("resume_wait_s", 60))
        if STOP.is_set() or over_wall():  # checked before every spawn, not only while one runs (Codex 24 Oct)
            log(f"not starting {name} (finalize or wall-clock limit)")
            return "timeout"
        proc = run_cmd(args, name)
        log(f"{name} running (attempt {attempt}): {' '.join(a if len(a) < 200 else a[:60] + '...' for a in args)}")
        while proc.poll() is None:
            time.sleep(CFG.get("poll_s", 30))
            with LOCK:
                state["progress"] = progress()
            save()
            if STOP.is_set() or over_wall():
                log(f"stopping {name} (finalize or wall-clock limit)")
                kill(proc)
                return "timeout"
            if VLLM is not None and VLLM.poll() is not None:
                kill(proc)
                raise RuntimeError(f"vllm died during {name}")
        if proc.returncode == 0:
            return "done"
        left = _missing(out) if out is not None else {}
        size = sum(v for v in left.values() if isinstance(v, int))
        work = _work_done(out) if out is not None else 0
        progressed = before is None or size < before or work > work_before
        stalls = 0 if progressed else stalls + 1
        before, work_before = size, work
        log(f"{name} exited with {proc.returncode}; missing {left}; resuming")
        if "budget_stop" in left or stalls >= stalls_max:
            break  # a spend stop is not retried; neither is a run that stopped making progress
    log(f"{name} still incomplete: error (results are synced; a pod has no cross-pod resume: relaunch under a new "
        "out-name if needed)")
    return "error"


def install_harness():
    import importlib.util
    missing = [pkg for mod, pkg in (("scipy", "scipy"), ("dotenv", "python-dotenv"), ("numpy", "numpy"),
                                    ("openai", "openai")) if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    log(f"harness dependencies ok (installed: {missing or 'none'})")


def fallback():
    RUNS.mkdir(parents=True, exist_ok=True)
    serve()
    set_phase("error")
    log("fallback mode: main runner crashed")
    deadline = time.time() + CFG["ack_wait_min"] * 60
    while time.time() < deadline and not state["acked"]:
        time.sleep(15)
    self_end(state["acked"])
    if POD:  # keep the process alive while RunPod deletes the pod (not needed in a local smoke run)
        time.sleep(600)


def main():
    global WALL_H, T0
    RUNS.mkdir(parents=True, exist_ok=True)
    if STATE_F.exists():
        try:
            old = json.loads(STATE_F.read_text())
            state["log"] = old.get("log", [])
            state["restarts"] = old.get("restarts", 0) + 1
            if ON_MODAL and old.get("t0"):
                T0 = float(old["t0"])
        except Exception:
            pass
    if ON_MODAL:
        state["t0"] = T0
    serve()
    log(f"honeypot runner up for {CFG['model']}")
    rate = CFG.get("cost_per_hr") if ON_MODAL else own_cost_per_hr()
    WALL_H = min(CFG["wall_hours"], CFG["budget_usd"] / rate) if rate else CFG["wall_hours"]
    with LOCK:
        state["cost_per_hr"], state["wall_hours"] = rate, round(WALL_H, 2)
    log(f"cost {rate} $/h -> wall-clock limit {WALL_H:.2f} h (budget ${CFG['budget_usd']})")
    phase = "error"
    try:
        install_harness()
        phase = play()
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
    finally:
        stop_vllm()
    set_phase(phase)
    if ON_MODAL:
        return phase
    deadline = time.time() + CFG["ack_wait_min"] * 60
    while time.time() < deadline and not state["acked"]:
        time.sleep(15)
    self_end(state["acked"])
    if POD:  # keep the process alive while RunPod deletes the pod (not needed in a local smoke run)
        time.sleep(600)


if __name__ == "__main__":
    fallback() if "--fallback" in sys.argv else main()
