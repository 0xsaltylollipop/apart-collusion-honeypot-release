"""Pod-path fixes from Astra's re-check of 207886c (24 Oct): one launch per out-name at a time on either backend
(the check -> create -> record steps hold a laptop-side OS lock), and the pod's resume loop counts progress inside
a debrief (new answers, failed tries, give-up markers), not only the shrinking missing counts."""

import importlib.util
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POD_DIR = ROOT / "scripts" / "pod"
sys.path.insert(0, str(POD_DIR))
import modal_pod  # noqa: E402
import pod  # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the launch lock

def test_a_second_launch_while_the_first_is_between_check_and_record_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("locktest")
    created = []

    def body(a):  # stands in for check -> create pod -> save record
        created.append(1)
        with pytest.raises(SystemExit, match="in progress"):
            pod.launch(a)  # an overlapping launcher of the same run
        with pytest.raises(SystemExit, match="in progress"):
            modal_pod.launch(type("A", (), {"dry_run": False})())  # also on the other backend
        pod.save({"id": "p1", "created": 0})

    monkeypatch.setattr(pod, "_launch", body)
    pod.launch(None)
    assert created == [1]
    pod.launch(None)  # the lock is released afterwards: a later launch gets past it (and the record check stops it)
    assert created == [1, 1]


def test_the_lock_holds_across_processes(tmp_path):
    child = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(POD_DIR)!r})
        import pod
        pod.MAIN = __import__("pathlib").Path({str(tmp_path)!r})
        pod.set_out("xproc")
        with pod.launch_lock():
            print("held", flush=True)
            time.sleep(5)
    """)
    p = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        old = pod.MAIN
        try:
            pod.MAIN = tmp_path
            pod.set_out("xproc")
            with pytest.raises(SystemExit, match="in progress"):
                with pod.launch_lock():
                    pass
        finally:
            pod.MAIN = old
    finally:
        p.kill()
        p.wait()
    time.sleep(0.1)
    pod.MAIN = tmp_path
    try:
        pod.set_out("xproc")
        with pod.launch_lock():  # the dead holder's lock is gone with its process (never stale)
            pass
    finally:
        pod.MAIN = old


def test_a_live_record_on_either_backend_blocks_a_new_launch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("live")
    monkeypatch.setattr(pod, "run_cfg", lambda a: (_ for _ in ()).throw(AssertionError("must not get this far")))
    monkeypatch.setattr(modal_pod, "modal_cfg", lambda a: (_ for _ in ()).throw(AssertionError("nor this")))
    a = type("A", (), {"dry_run": False, "gpu": "L40S"})()
    pod.OUT.mkdir(parents=True)
    (pod.OUT / "modal.json").write_text(json.dumps({"app_id": "ap-1"}))  # Modal run not ended
    pod.launch(a)
    assert "modal.json records a launched run" in capsys.readouterr().out
    (pod.OUT / "modal.json").write_text(json.dumps({"app_id": "ap-1", "ended": 1.0}))
    pod.save({"id": "pod-1", "created": 0})  # RunPod pod not ended
    modal_pod.launch(a)
    assert "pod.json records a launched run" in capsys.readouterr().out


def test_a_modal_dry_run_still_takes_no_lock_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("dry2")
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    pod.add_launch_args(ap)
    ap.add_argument("--gpu", default=modal_pod.DEFAULT_GPU)
    a = ap.parse_args(["--hf-model", "openai/gpt-oss-20b", "--served", "gpt-oss-20b", "--revision", "abc",
                       "--parser", "openai_gptoss", "--trap", json.dumps({"games": 1}), "--dry-run"])
    modal_pod.launch(a)
    assert "-m modal run --detach" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()


# ---------------------------------------------------------------- progress inside a debrief

class _Proc:
    def __init__(self, rc):
        self.returncode = rc

    def poll(self):
        return self.returncode


def _runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HP_CFG", json.dumps({"model": "m", "served": "gpt-oss-20b", "mode": "trap",
                                             "trap": {"games": 1, "blind": True}, "poll_s": 0, "resume_wait_s": 0,
                                             "max_tokens": 32768, "max_model_len": 65536, "concurrency": 16}))
    monkeypatch.setenv("RESULTS_TOKEN", "t")
    monkeypatch.setenv("POD_WORKSPACE", str(tmp_path))
    return _module("runner_lock_test", POD_DIR / "runner.py")


def test_debrief_progress_without_a_shrinking_count_keeps_the_pod_resuming(tmp_path, monkeypatch):
    """Astra's repro: two transient failures, then subject A answers and B fails once; the missing count stays
    {"debriefs": 1} throughout, but each resume logs new debrief records, so the runner keeps going to completion."""
    r = _runner(tmp_path, monkeypatch)
    out = tmp_path / "o"
    deb = out / "copies" / "trunk-s1@k10" / "offer-b0" / "debrief.jsonl"
    deb.parent.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({"incomplete": {"debriefs": 1}}))
    script = iter([None, None, {"firm": 0, "q": 1, "answer": "ok"}, {"firm": 1, "q": 1, "failed": True},
                   {"firm": 1, "q": 1, "failed": True}, "done"])
    calls = []

    def fake(args, name):
        step = next(script)
        calls.append(step)
        if step == "done":
            return _Proc(0)
        if step is not None:
            with open(deb, "a") as f:
                f.write(json.dumps(step) + "\n")
        return _Proc(3)

    monkeypatch.setattr(r, "run_cmd", fake)
    # attempts 1-2: transient (nothing logged; the first attempt never counts as a stall, the second is stall 1);
    # 3-5 each log a record (progress); 6 completes. The old missing-count rule stopped after attempt 3.
    assert r._run_until_complete(["run"], "pilot", out) == "done"
    assert len(calls) == 6


def test_no_new_records_and_no_shrinking_count_still_stops(tmp_path, monkeypatch):
    r = _runner(tmp_path, monkeypatch)
    out = tmp_path / "o"
    (out / "copies").mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({"incomplete": {"debriefs": 1}}))
    n = {"v": 0}
    monkeypatch.setattr(r, "run_cmd", lambda args, name: n.__setitem__("v", n["v"] + 1) or _Proc(3))
    assert r._run_until_complete(["run"], "pilot", out) == "error"
    assert n["v"] == 3  # the first attempt, then 2 stalled resumes


# ---------------------------------------------------------------- Codex on cfe75dc

def test_the_modal_record_exists_while_the_app_is_starting(tmp_path, monkeypatch):
    """An interrupted launcher must leave a live record (else a retry starts a second App on the same folder)."""
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("intent")
    monkeypatch.setattr(modal_pod, "modal_cfg", lambda a: {"cost_per_hr": 4.6, "budget_usd": 50, "wall_hours": 8,
                                                         "git_commit": "c", "served": "gpt-oss-20b",
                                                         "trap": {"c_model": "x"}})
    monkeypatch.setattr(modal_pod, "key_names", lambda cfg: [])
    monkeypatch.setattr(pod, "prereg_text", lambda cfg: "plan\n")

    class P:
        pid = 4242

        def poll(self):
            return None
    monkeypatch.setattr(modal_pod.subprocess, "Popen", lambda *a, **k: P())
    seen = []

    def interrupt(_):
        seen.append(modal_pod.load())
        raise KeyboardInterrupt  # Ctrl-C while waiting for the App id
    monkeypatch.setattr(modal_pod.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        modal_pod.launch(type("A", (), {"dry_run": False, "gpu": "H100"})())
    assert seen[0]["phase"] == "launching" and seen[0]["client_pid"] == 4242
    assert pod.live_record()[0] == "modal.json"  # a retry is refused


def test_a_modal_app_stopped_while_queued_ends_its_record(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("queued")
    modal_pod.save({"app_id": "ap-q", "created": time.time() - 3 * 3600, "cost_per_hr": 4.5, "budget_usd": 10,
                    "wall_hours": 8, "client_pid": 1, "stop_confirmed": True, "stop_reason": "queue"})
    import subprocess as sp  # noqa: PLC0415
    real = sp.run
    listing = {"ok": False}

    def fake(cmd, **kw):
        if "modal" not in cmd:
            return real(cmd, **kw)
        if "ls" in cmd:  # the Volume root: other runs' folders, not this one (or the listing itself fails)
            return sp.CompletedProcess(cmd, 0, "other-run\n", "") if listing["ok"] else sp.CompletedProcess(cmd, 1, "", "net")
        return sp.CompletedProcess(cmd, 1, "", "No such file or directory: queued/runs")
    monkeypatch.setattr(modal_pod.subprocess, "run", fake)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: False)
    # a failed download plus a failed listing is no evidence of "no output": the record stays live (Codex 24 Oct)
    assert not modal_pod.fetch_once().get("ended")
    listing["ok"] = True
    r = modal_pod.fetch_once()
    assert r.get("ended") and "before any container" in r["phase"]
    assert pod.live_record() is None


def test_a_stopped_run_with_output_on_the_volume_is_never_ended_by_a_download_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("hasout")
    modal_pod.save({"app_id": "ap-o", "created": time.time() - 3 * 3600, "cost_per_hr": 4.5, "budget_usd": 10,
                    "wall_hours": 8, "client_pid": 1, "stop_confirmed": True, "stop_reason": "budget"})
    import subprocess as sp  # noqa: PLC0415
    real = sp.run

    def fake(cmd, **kw):
        if "modal" not in cmd:
            return real(cmd, **kw)
        if "ls" in cmd:
            return sp.CompletedProcess(cmd, 0, "hasout\n" if cmd[-1] == "/" else "hasout/runs\n", "")
        return sp.CompletedProcess(cmd, 1, "", "auth error")  # results exist but the download fails
    monkeypatch.setattr(modal_pod.subprocess, "run", fake)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: False)
    assert not modal_pod.fetch_once().get("ended")


def test_a_non_capacity_runpod_error_is_not_retried_on_the_other_cloud(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("http500")
    monkeypatch.setattr(pod, "run_cfg", lambda a: {"wall_hours": 1, "ack_wait_min": 1, "budget_usd": 3,
                                                   "served": "gpt-oss-20b", "trap": {}})
    monkeypatch.setattr(pod, "git_commit", lambda: "c")
    monkeypatch.setattr(pod, "prereg_text", lambda cfg: "plan\n")
    monkeypatch.setattr(pod, "bundle_b64", lambda: "x")
    monkeypatch.setattr(pod, "pod_env", lambda cfg, token, chunks: {})
    clouds = []

    def rp(method, path, body=None):
        clouds.append(body["cloudType"])
        raise RuntimeError("RunPod POST /pods: HTTP 500 b'internal'")
    monkeypatch.setattr(pod, "rp", rp)
    pod.launch(None)
    assert clouds == ["COMMUNITY"] and "not a capacity error" in capsys.readouterr().out
    assert pod.load() is None
