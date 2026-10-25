#!/usr/bin/env python3
"""Self-terminating RunPod pods for the honeypot run and self-play: the run happens ON the GPU pod, which ends itself;
the laptop only launches the pod and fetches results.

    python scripts/pod/pod.py --out-name <name> launch --hf-model <repo> --served <id> --revision <sha> --parser <p> \
        --trap '{"games": 6, "ks": [10], "copy_len": 30, "reps": 1, "selfplay": [...]}' [--dry-run]
    python scripts/pod/pod.py --out-name <name> fetch [--loop S]   # pull status + results; ack a finished pod
    python scripts/pod/pod.py --out-name <name> status             # pod state + accrued cost
    python scripts/pod/pod.py --out-name <name> down               # emergency: terminate THIS run's pod only
    scripts/pod/models.sh <model> ...                              # the served models and their exact settings

The code bundled onto the pod is the collusion package of this checkout. Keys come from this repo's .env (OpenAI keys
only when the subject or firm C is an OpenAI model); results land in runs/<out-name>/ (git-ignored). Secrets are never
printed.
"""

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import secrets
import shutil
import subprocess
import tarfile
import tempfile
import time
import tokenize
import urllib.error
import urllib.request
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]  # this checkout: its collusion/ package is shipped to the pod
MAIN = Path(os.environ.get("HP_MAIN_ROOT", str(CODE)))  # where .env and runs/ live
OUT = MAIN / "runs" / "unnamed"  # set by --out-name
POD_F = OUT / "pod.json"
REST = "https://rest.runpod.io/v1"
UA = "apart-collusion-honeypot/0.1"
IMAGE = "vllm/vllm-openai:v0.31.0"

CFG = dict(max_model_len=32768, max_tokens=4096, concurrency=96, budget_usd=10.0, wall_hours=6.0, ack_wait_min=30)
LAPTOP_GRACE_MIN = 10  # after the final ack, hp.py terminates the pod itself if it is still listed
# default list for a direct pod.py launch; the main runs go through models.sh, which passes its own GPU80 list (the
# same 80 GB class as the self-play pods). 24 Oct: the smoke test's capacity problem was this short list only.
GPUS = ["NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"]


def set_out(name):
    global OUT, POD_F
    OUT = MAIN / "runs" / name
    POD_F = OUT / "pod.json"


def llm_module():
    """collusion.clients.llm of the checkout that is shipped to the pod (for its model router)."""
    sys.path.insert(0, str(CODE))
    from collusion.clients import llm  # noqa: PLC0415
    return llm


def run_cfg(a) -> dict:
    """CFG for this launch: the 18 Oct pilot defaults, updated from the launch flags."""
    cfg = dict(CFG)
    for key in ("served", "revision", "parser", "budget_usd", "wall_hours", "max_tokens", "max_model_len", "gpus",
                "disk_gb", "concurrency"):
        val = getattr(a, key, None)
        if val is not None:
            cfg[key] = val
    if getattr(a, "hf_model", None):
        cfg["model"] = a.hf_model
    if getattr(a, "chat_template_kwargs", None):
        cfg["chat_template_kwargs"] = json.loads(a.chat_template_kwargs)
    if getattr(a, "extra_vllm", None):
        cfg["extra_vllm"] = json.loads(a.extra_vllm)
    if not getattr(a, "trap", None):
        raise SystemExit("--trap is required (the honeypot run and/or self-play spec)")
    cfg["mode"], cfg["trap"] = "trap", json.loads(a.trap)  # python -m collusion run / selfplay on the pod
    cfg["trap"].setdefault("c_model", "gpt-5.4-mini-2026-03-17")
    check_sub_schedule(cfg["trap"])
    return cfg


def check_sub_schedule(t: dict):
    """The k=5 subset (sub_ks in the first sub_games games) checked on the laptop, before a pod is paid for: the pod's
    run would refuse a bad schedule only after booting (review 24 Oct)."""
    sub_ks, sub_games, ks = t.get("sub_ks") or [], t.get("sub_games") or 0, t.get("ks", [10])
    if not sub_ks and not sub_games:
        return
    ok = (isinstance(sub_ks, list) and sub_ks and all(type(k) is int and k >= 1 for k in sub_ks)
          and len(set(sub_ks)) == len(sub_ks) and not set(sub_ks) & set(ks) and max(sub_ks) < max(ks)
          and type(sub_games) is int and 0 < sub_games <= t.get("games", 2))
    if not ok:
        raise SystemExit(f"bad k=5 subset schedule: sub_ks {sub_ks!r}, sub_games {sub_games!r} (need distinct ints "
                         f"below the last main save point {max(ks)} and not among ks {ks}, and 0 < sub_games <= games)")


def prereg_text(cfg: dict) -> str:
    """Pre-registration line for RESULTS.md, written before any data arrives."""
    return ("## Honeypot / self-play run\n\nRules and measures: PREREG.md and docs/DECISIONS.md at commit "
            f"{git_commit()}.\n"
            f"Setup: subject {cfg['model']} (served as {cfg['served']}, revision {cfg.get('revision')}), run "
            f"{json.dumps(cfg['trap'])}, max_model_len {cfg['max_model_len']}, max_tokens {cfg['max_tokens']}, "
            f"chat template kwargs {cfg.get('chat_template_kwargs')}.\n")


def env():
    vals = {}
    for line in (MAIN / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    return vals


def rp(method, path, body=None, timeout=60):
    req = urllib.request.Request(f"{REST}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Authorization", f"Bearer {env()['RUNPOD_API_KEY']}")
    req.add_header("User-Agent", UA)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RunPod {method} {path}: HTTP {e.code} {e.read()[:300]!r}") from None


def load():
    return json.loads(POD_F.read_text()) if POD_F.exists() else None


def save(p):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = POD_F.with_suffix(".tmp")
    tmp.write_text(json.dumps(p, indent=2))
    tmp.replace(POD_F)


def pod_http(p, method, path, timeout=60, raw=False):
    req = urllib.request.Request(f"https://{p['id']}-8001.proxy.runpod.net{path}", method=method,
                                 data=b"{}" if method == "POST" else None)
    req.add_header("Authorization", f"Bearer {p['token']}")
    req.add_header("User-Agent", UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return data if raw else json.loads(data)


# Modules the pod never imports: the cross-model analysis, the words judge and the robot check run on the laptop.
# collusion/cli.py imports them only when they are present, so `python -m collusion run / selfplay / analyze` work
# without them (tests/test_pod_bundle.py runs those entry points from an extracted bundle).
LAPTOP_ONLY = {"collusion/analysis/crossmodel.py", "collusion/analysis/judge.py", "collusion/experiments/robots.py",
               # the words judge's analysis side (24 Oct, analysis-v4): read on the laptop only, never on a pod
               "collusion/analysis/judge_selfplay.py", "collusion/analysis/words.py", "collusion/analysis/verdicts.py",
               # the Opus majority pseudo-judge (25 Oct): analysis on the laptop only
               "collusion/analysis/majority.py"}
# RunPod answers HTTP 500 above roughly 100 KB of pod env (19 Oct: ~110 KB failed; 24 Oct: ~103 KB failed, ~91 KB
# went through), so the base64 bundle must stay well below this (checked on every build and in the test).
BUNDLE_LIMIT_B64 = 85_000


def strip_comments(src: str) -> str:
    """`src` without its # comments, every line kept in place, so a pod traceback has the laptop's line numbers."""
    lines = io.StringIO(src).readlines()  # the same line split as tokenize's (splitlines also breaks at \f, \x1c, ...)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            r, c = tok.start
            nl = "\n" if lines[r - 1].endswith("\n") else ""
            lines[r - 1] = lines[r - 1][:c].rstrip() + nl
    return "".join(lines)


def _docstring_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value,
                                                                                                   str):
                yield first


def strip_docstrings(src: str) -> str:
    """`src` with every docstring replaced by "" in place, every line kept (24 Oct evening: the bundle had grown to
    ~81 KB of base64 again). Nothing in the shipped code reads __doc__; the AST check in bundle_b64 compares the
    result with the original's AST with its docstrings emptied, so only docstrings can differ."""
    lines = [ln.encode() for ln in io.StringIO(src).readlines()]  # ast columns are UTF-8 byte offsets
    for e in sorted(_docstring_nodes(ast.parse(src)), key=lambda e: (e.lineno, e.col_offset), reverse=True):
        l1, c1, l2, c2 = e.lineno - 1, e.col_offset, e.end_lineno - 1, e.end_col_offset
        tail = lines[l2][c2:]
        if l1 == l2:
            lines[l1] = lines[l1][:c1] + b'""' + tail
        elif tail.strip():  # code after a multi-line docstring on its last line ('"""; x = 1'): left as it is
            continue
        else:
            lines[l1] = lines[l1][:c1] + b'""\n'
            for j in range(l1 + 1, l2):
                lines[j] = b"\n"
            lines[l2] = tail  # "\n", or "" at the end of a file without a final newline (Codex 24 Oct)
    return b"".join(lines).decode()


def _without_docstrings(tree, src: str | None = None):
    """The AST with its docstrings emptied, as strip_docstrings leaves them (with src: except a multi-line
    docstring followed by code on its last line, which is kept)."""
    lines = [ln.encode() for ln in io.StringIO(src).readlines()] if src is not None else None
    for e in _docstring_nodes(tree):
        if lines is not None and e.lineno != e.end_lineno and lines[e.end_lineno - 1][e.end_col_offset:].strip():
            continue
        e.value.value = ""
    return tree


def bundle_b64() -> str:
    """The pod's code as base64 for the pod env: the collusion package without LAPTOP_ONLY and caches, its comments
    stripped (24 Oct: the gz tar had grown to ~121 KB of base64; bz2 and the exclusions alone left ~83 KB, too close to
    the limit), plus pyproject.toml, runner.py and selfstop.py; a bz2 tar, unpacked on the pod with mode "r:*". Every
    stripped file's AST is checked to equal the original's, so the pod runs exactly this checkout's code."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        for f in sorted((CODE / "collusion").rglob("*")):  # data files too (clients/prices.json)
            rel = f.relative_to(CODE).as_posix()
            if f.is_dir() or "__pycache__" in rel or rel.endswith(".pyc") or rel in LAPTOP_ONLY:
                continue
            data = f.read_bytes()
            if f.suffix == ".py":
                src = data.decode()
                bare = strip_docstrings(strip_comments(src))
                nocom = strip_comments(src)  # the docstring rule is judged on the comment-stripped text (Codex)
                if (ast.dump(ast.parse(bare)) != ast.dump(_without_docstrings(ast.parse(src), nocom))
                        or bare.count("\n") != src.count("\n")):
                    raise RuntimeError(f"comment/docstring stripping changed the code or line count of {rel}")
                data = bare.encode()
            ti = tarfile.TarInfo(f"code/{rel}")
            ti.size, ti.mode, ti.mtime = len(data), 0o644, int(f.stat().st_mtime)
            tf.addfile(ti, io.BytesIO(data))
        tf.add(CODE / "pyproject.toml", arcname="code/pyproject.toml")
        for f in ("runner.py", "selfstop.py"):
            tf.add(Path(__file__).parent / f, arcname=f)
    b64 = base64.b64encode(buf.getvalue()).decode()
    if len(b64) > BUNDLE_LIMIT_B64:
        raise SystemExit(f"pod code bundle is {len(b64)} bytes of base64, over the {BUNDLE_LIMIT_B64} limit (RunPod "
                         "answers HTTP 500 to a large pod env): add laptop-only modules to LAPTOP_ONLY")
    return b64


def git_commit():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "collusion"], cwd=CODE, capture_output=True,
                           text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


START = (
    "mkdir -p /workspace/boot /workspace/runs && cd /workspace/boot && "
    "python3 -c 'import os,base64,io,tarfile; "
    "d=base64.b64decode(\"\".join(os.environ[\"HARNESS_B64_%d\" % i] for i in range(int(os.environ[\"HARNESS_CHUNKS\"])))); "
    "tarfile.open(fileobj=io.BytesIO(d),mode=\"r:*\").extractall(\".\")'; "
    "( sleep {hard}; python3 /workspace/boot/selfstop.py ) >> /workspace/runs/backstop.log 2>&1 & "
    "python3 /workspace/boot/runner.py >> /workspace/runs/runner.log 2>&1 || "
    "python3 /workspace/boot/runner.py --fallback >> /workspace/runs/runner.log 2>&1; "
    "sleep infinity"
)


def pod_env(cfg: dict, token: str, chunks: list[str]) -> dict:
    """Pod environment. The OpenAI keys (the second one is the fallback when the first runs out) are included only
    when the subject or firm C is an OpenAI model."""
    llm_mod = llm_module()
    envs = {"HP_CFG": json.dumps(cfg), "RESULTS_TOKEN": token, "HARNESS_CHUNKS": str(len(chunks)),
            "SELF_KEY": env()["RUNPOD_API_KEY"]}
    if any(m and llm_mod.route(m) == "openai" for m in (cfg.get("served"), (cfg.get("trap") or {}).get("c_model"))):
        for k in llm_mod.KEYS["openai"]:
            if env().get(k):
                envs[k] = env()[k]
    envs.update({f"HARNESS_B64_{i}": c for i, c in enumerate(chunks)})
    return envs


LAUNCH_LOCK = ".launch.lock"


@contextlib.contextmanager
def launch_lock():
    """An exclusive laptop-side lock on this out-name, held from the "already launched?" check through creating the
    pod (or Modal App) to saving its record (Astra 24 Oct: two overlapping launches could both create a pod, and the
    last record written won, leaving the other untracked). pod.py and modal_pod.py share it, so one run can't be
    launched on both backends at once either. An OS lock: it goes away with a crashed launcher, never stale."""
    import fcntl  # noqa: PLC0415
    OUT.mkdir(parents=True, exist_ok=True)
    f = open(OUT / LAUNCH_LOCK, "a+")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another launch of {OUT.name} is in progress (lock {OUT / LAUNCH_LOCK}); "
                             "not launching") from None
        yield
    finally:
        f.close()


def live_record():
    """This out-name's launched-and-not-ended record on either backend (RunPod pod.json, Modal modal.json)."""
    for f in (POD_F, OUT / "modal.json"):
        if f.exists():
            r = json.loads(f.read_text())
            if not r.get("ended"):
                return f.name, r
    return None


def launch(a):
    with launch_lock():
        _launch(a)


def _launch(a):
    live = live_record()
    if live:
        print(f"{live[0]} records a launched run that has not ended ({live[1].get('id') or live[1].get('app_id')}); "
              "not launching another")
        return
    cfg = {**run_cfg(a), "git_commit": git_commit()}
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "RESULTS.md").exists():  # the pre-registered plan goes in before any data arrives
        (OUT / "RESULTS.md").write_text(prereg_text(cfg)
                                        + f"\nCode commit at launch: {git_commit()}\n\n## Results\n\n(to be filled after the run)\n")
    b64 = bundle_b64()
    print(f"pod code bundle: {len(b64)} bytes of base64 (limit {BUNDLE_LIMIT_B64})", flush=True)
    chunks = [b64[i:i + 8000] for i in range(0, len(b64), 8000)]
    token = secrets.token_urlsafe(24)
    hard = int((cfg["wall_hours"] + cfg["ack_wait_min"] / 60 + 0.75) * 3600)
    envs = pod_env(cfg, token, chunks)
    body = {"name": f"collusion-{OUT.name}"[:60], "imageName": IMAGE, "gpuTypeIds": cfg.get("gpus") or GPUS, "gpuCount": 1,
            "allowedCudaVersions": ["13.0"], "containerDiskInGb": cfg.get("disk_gb") or 60, "volumeInGb": 20,
            "volumeMountPath": "/workspace", "ports": ["8001/http"], "env": envs,
            "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [START.format(hard=hard)]}
    pod = cloud = None
    other = False  # any failure that is not RunPod's "no instances" (it may have created a pod anyway)
    for cloud in ("COMMUNITY", "SECURE"):
        try:
            pod = rp("POST", "/pods", {**body, "cloudType": cloud})
            break
        except RuntimeError as e:
            print(f"{cloud} failed: {str(e)[:160]}")
            if not any(s in str(e) for s in ("no instances currently available",
                                             "no longer any instances available")):  # RunPod's two wordings
                # anything else (HTTP 500, a timeout) may have created a pod anyway: never try the other cloud
                # type on top of it (Codex 24 Oct: both could exist, only the second recorded)
                other = True
                break
    if not pod:
        print("launch failed (not a capacity error); check the RunPod console" if other else "no capacity; not launched")
        return
    save({"id": pod["id"], "token": token, "created": time.time(), "cost_per_hr": pod.get("costPerHr"),
          "cloud": cloud, "gpu": (pod.get("machine") or {}).get("gpuTypeId") or (pod.get("gpu") or {}).get("id"),
          "gpu_types_allowed": cfg.get("gpus") or GPUS, "budget_usd": cfg["budget_usd"], "wall_hours": cfg["wall_hours"], "phase": "created",
          "git_commit": cfg["git_commit"], "served": cfg["served"], "c_model": cfg["trap"].get("c_model")})
    print(f"pod {pod['id']} {cloud} {pod.get('costPerHr')} $/h (budget ${cfg['budget_usd']})")


def accrued(p, now=None):
    end = p.get("ended") or now or time.time()
    return (p.get("cost_per_hr") or 0) * (end - p["created"]) / 3600


def extract(data):
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(td, filter="data")
        place(Path(td) / "runs")


def place(src: Path):
    """Move a pod's runs/ folder into runs/<out-name>/: honeypot/* at the top, everything else (state, logs) under
    _pod/. Shared with the Modal path (modal_pod.py), whose runs/ folder comes from the Volume."""
    OUT.mkdir(parents=True, exist_ok=True)
    podmeta = OUT / "_pod"
    podmeta.mkdir(exist_ok=True)
    for item in src.iterdir():
        target = OUT if item.name == "honeypot" else podmeta
        if item.name == "honeypot":
            for g in item.iterdir():
                t = OUT / g.name
                if t.exists():
                    subprocess.run(["rm", "-rf", str(t)], check=True)
                shutil.move(str(g), str(t))  # works across filesystems (rename does not)
        else:
            t = target / item.name
            if t.exists():
                subprocess.run(["rm", "-rf", str(t)], check=True)
            shutil.move(str(item), str(t))


def fetch_once():
    p = load()
    if not p:
        print("no pilot pod recorded")
        return None
    if p.get("ended"):
        print(f"pod {p['id']} ended ({p.get('phase')}); cost ${accrued(p):.2f}")
        return p
    live = {q["id"]: q for q in (rp("GET", "/pods") or [])}
    q = live.get(p["id"])
    now = time.time()
    hard_h = p.get("wall_hours", CFG["wall_hours"]) + CFG["ack_wait_min"] / 60 + 1.0
    if q is not None and (
            (p.get("final_fetched") and now - p.get("acked_at", now) > LAPTOP_GRACE_MIN * 60)
            or q.get("desiredStatus") == "EXITED"
            or accrued(p, now) > p["budget_usd"] * 1.5
            or now - p["created"] > hard_h * 3600):
        print(f"laptop-side fallback: terminating pod {p['id']} (status {q.get('desiredStatus')})")
        try:
            rp("DELETE", f"/pods/{p['id']}")
        except RuntimeError as e:
            print(f"terminate failed: {e}")
        q = None
    if q is None:
        p["ended"], p["phase"] = now, f"{p.get('phase')} (pod gone)"
    else:
        try:
            st = pod_http(p, "GET", "/status", timeout=30)
            p["phase"], p["progress"] = st["phase"], st.get("progress")
            final = st["phase"] in ("done", "timeout", "error")
            data = pod_http(p, "GET", "/results.tgz", timeout=600, raw=True)
            extract(data)
            p["last_fetch"] = time.time()
            if final and not p.get("final_fetched"):
                p["final_fetched"] = bool(pod_http(p, "POST", "/ack", timeout=30).get("acked"))
                p["acked_at"] = time.time()
            elif accrued(p) > p["budget_usd"] * 1.05 and not p.get("finalize_sent"):
                pod_http(p, "POST", "/finalize", timeout=30)
                p["finalize_sent"] = True
        except Exception as e:
            p["last_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    save(p)
    print(f"{time.strftime('%H:%M:%S')} pod {p['id']} phase={p.get('phase')} progress={p.get('progress')} "
          f"cost ${accrued(p):.2f}/{p['budget_usd']} err={p.get('last_error', '')}")
    return p


def fetch(a):
    while True:
        try:
            p = fetch_once()
        except Exception as e:  # 21 Oct: a DNS failure while the laptop changed networks killed two loops
            if not a.loop:
                raise
            print(f"{time.strftime('%H:%M:%S')} fetch failed ({type(e).__name__}: {str(e)[:120]}); retrying",
                  flush=True)
            time.sleep(a.loop)
            continue
        if not a.loop or not p or p.get("ended"):
            return
        time.sleep(a.loop)


def status(_):
    p = load()
    for q in rp("GET", "/pods") or []:
        mine = " (this pilot)" if p and q["id"] == p["id"] else ""
        print(q["id"], q.get("name"), q.get("desiredStatus"), q.get("costPerHr"), "$/h" + mine)
    if p:
        print(f"pilot pod {p['id']}: phase={p.get('phase')} cost ${accrued(p):.2f}")


def down(_):
    p = load()
    if not p:
        print("no pilot pod recorded")
        return
    try:
        rp("DELETE", f"/pods/{p['id']}")
        print(f"terminated {p['id']}")
    except RuntimeError as e:
        print(f"terminate failed: {e}")
    p["ended"] = p.get("ended") or time.time()
    save(p)
    print("remaining pods on the account:", [q["id"] for q in rp("GET", "/pods") or []])


def add_launch_args(la):
    """The launch flags (shared with modal_pod.py, so a Modal launch takes exactly the same model settings)."""
    la.add_argument("--hf-model", required=True, help="HF repo of the subject model")
    la.add_argument("--served", help="served model id (must route to vLLM, e.g. qwen3.5-4b)")
    la.add_argument("--revision")
    la.add_argument("--parser", help="vLLM reasoning parser (deepseek_r1, qwen3, ...)")
    la.add_argument("--chat-template-kwargs", help='JSON, e.g. \'{"enable_thinking": false}\'')
    la.add_argument("--extra-vllm", help='JSON list of extra vllm serve flags, e.g. \'["--language-model-only"]\'')
    la.add_argument("--budget-usd", type=float)
    la.add_argument("--max-tokens", type=int, help="per-call generation cap (default 4096; thinking models need more)")
    la.add_argument("--max-model-len", type=int, help="vLLM context length (default 32768)")
    la.add_argument("--gpus", type=json.loads, help='JSON list of RunPod GPU type ids, in order of preference')
    la.add_argument("--wall-hours", type=float)
    la.add_argument("--concurrency", type=int, help="games/copies in flight at once (default 96); vLLM streams = "
                    "this x LLM firms per game (self-play 3, honeypot 2). docs/pod-throughput-lessons.md")
    la.add_argument("--disk-gb", type=int, help="container disk (default 60; BF16 ~30B weights need ~150)")
    la.add_argument("--trap", required=True, help='JSON: the honeypot run (games 0 = self-play only) and its '
                    '"selfplay" scenarios, e.g. \'{"games": 2, "ks": [10], "copy_len": 30, "reps": 1, "c_effort": "low"}\'')
    la.add_argument("--dry-run", action="store_true", help="print the run config and pre-registration, launch nothing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", required=True, help="results go to runs/<out-name>/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch")
    add_launch_args(la)
    f = sub.add_parser("fetch")
    f.add_argument("--loop", type=int, default=0)
    sub.add_parser("status")
    sub.add_parser("down")
    a = ap.parse_args()
    set_out(a.out_name)
    if a.cmd == "launch" and a.dry_run:
        cfg = run_cfg(a)
        print(json.dumps({k: v for k, v in cfg.items()}, indent=2))
        print(prereg_text(cfg)[:1500])
        return
    {"launch": launch, "fetch": fetch, "status": status, "down": down}[a.cmd](a)


if __name__ == "__main__":
    main()
