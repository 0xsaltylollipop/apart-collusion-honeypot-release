"""The Modal fallback (scripts/pod/modal_app.py, modal_pod.py; 24 Oct, RunPod had no CUDA-13 hosts): it reuses pod.py's
launch flags and run config, the runner behaves on Modal as on a pod except for the rate, the restart clock and the
missing ack, and results land in the pod layout. Runs without the modal package and without a Modal login."""

import ast
import importlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POD_DIR = ROOT / "scripts" / "pod"
sys.path.insert(0, str(POD_DIR))
import modal_pod  # noqa: E402
import pod  # noqa: E402

LAUNCH = ["--hf-model", "openai/gpt-oss-20b", "--served", "gpt-oss-20b", "--revision", "abc", "--parser",
          "openai_gptoss", "--max-tokens", "32768", "--max-model-len", "65536", "--budget-usd", "20",
          "--wall-hours", "8", "--chat-template-kwargs", '{"enable_thinking": true}',
          "--trap", json.dumps({"games": 10, "ks": [10], "sub_ks": [5], "sub_games": 3, "copy_len": 30,
                                "seed0": 2001})]


def _args(*extra):
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    pod.add_launch_args(ap)
    ap.add_argument("--gpu", default=modal_pod.DEFAULT_GPU)
    return ap.parse_args([*LAUNCH, *extra])


def test_modal_app_uses_the_pod_image():
    tree = ast.parse((POD_DIR / "modal_app.py").read_text())
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert consts["IMAGE"] == pod.IMAGE


def test_modal_config_is_the_pod_config_plus_the_platform_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("mtest")
    a = _args("--gpu", "H100")
    base, cfg = pod.run_cfg(a), modal_pod.modal_cfg(a)
    extra = {k: v for k, v in cfg.items() if k not in base}
    assert {k: cfg[k] for k in base if k != "budget_usd"} == {k: v for k, v in base.items() if k != "budget_usd"}  # weights, revision, parser, template kwargs, sampling, run spec
    assert set(extra) == {"git_commit", "platform", "cost_per_hr", "budget_usd_pod"} and extra["platform"] == "modal"
    assert extra["budget_usd_pod"] == 20 and cfg["budget_usd"] == pytest.approx(8 * extra["cost_per_hr"], abs=0.01)
    assert extra["cost_per_hr"] == pytest.approx(3.95 + modal_pod.CPU_MEM_USD_H, abs=0.001)
    cmd, env = modal_pod.launch_cmd(cfg, "H100", tmp_path / "cfg.json")
    assert cmd[cmd.index("run") + 1] == "--detach" and cmd[cmd.index("--gpu") + 1] == "H100"
    assert float(cmd[cmd.index("--timeout-h") + 1]) <= 24 and env["HP_MODAL_APP"].startswith("collusion-mtest")
    assert "OPENAI_API_KEY" in cmd[cmd.index("--keys") + 1]  # firm C (gpt-5.4-mini) needs the OpenAI keys
    assert not any("sk-" in x for x in cmd)  # key names only, never values


def test_modal_refuses_unknown_gpus_and_over_long_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("mtest")
    with pytest.raises(SystemExit):
        modal_pod.modal_cfg(_args("--gpu", "A6000"))
    with pytest.raises(SystemExit):
        modal_pod.modal_cfg(_args("--wall-hours", "23.5"))


def test_place_gives_the_pod_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("ptest")
    src = tmp_path / "vol" / "runs"
    (src / "honeypot" / "trunks" / "trunk-s1").mkdir(parents=True)
    (src / "honeypot" / "trunks" / "trunk-s1" / "rounds.jsonl").write_text("{}\n")
    (src / "honeypot" / "manifest.json").write_text("{}")
    (src / "state.json").write_text(json.dumps({"phase": "done"}))
    (src / "runner.log").write_text("log\n")
    pod.place(src)
    out = tmp_path / "runs" / "ptest"
    assert (out / "trunks" / "trunk-s1" / "rounds.jsonl").exists() and (out / "manifest.json").exists()
    assert (out / "_pod" / "state.json").exists() and (out / "_pod" / "runner.log").exists()
    # pod.extract still does the same from a results.tgz
    src2 = tmp_path / "vol2" / "runs"
    (src2 / "honeypot").mkdir(parents=True)
    (src2 / "honeypot" / "x.json").write_text("1")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(src2, arcname="runs")
    pod.extract(buf.getvalue())
    assert (out / "x.json").read_text() == "1"


def _runner(monkeypatch, tmp_path, cfg):
    monkeypatch.setenv("POD_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("HP_CFG", json.dumps(cfg))
    monkeypatch.setenv("RESULTS_TOKEN", "t")
    monkeypatch.setenv("HARNESS_DIR", str(tmp_path / "code"))
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    sys.modules.pop("runner", None)
    runner = importlib.import_module("runner")
    monkeypatch.setattr(runner, "serve", lambda: None)
    monkeypatch.setattr(runner, "install_harness", lambda: None)
    monkeypatch.setattr(runner, "play", lambda: "done")
    monkeypatch.setattr(runner, "stop_vllm", lambda: None)
    return runner


CFG = {"model": "m", "served": "s", "wall_hours": 8.0, "budget_usd": 9.0, "ack_wait_min": 30}


def test_runner_on_modal_uses_the_config_rate_and_returns_without_an_ack(tmp_path, monkeypatch):
    runner = _runner(monkeypatch, tmp_path, {**CFG, "platform": "modal", "cost_per_hr": 4.5})
    monkeypatch.setattr(runner, "own_cost_per_hr", lambda: pytest.fail("RunPod API called on Modal"))
    t = time.time()
    assert runner.main() == "done" and time.time() - t < 5  # no 30-minute ack wait
    assert runner.WALL_H == pytest.approx(2.0)  # min(8 h, $9 / 4.5 $/h)
    assert runner.HARNESS == tmp_path / "code"
    st = json.loads((tmp_path / "runs" / "state.json").read_text())
    assert st["phase"] == "done" and st["t0"] == pytest.approx(runner.T0)


def test_runner_on_modal_keeps_the_first_start_after_a_restart(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "state.json").write_text(json.dumps({"t0": 1000.0, "restarts": 0, "log": []}))
    runner = _runner(monkeypatch, tmp_path, {**CFG, "platform": "modal", "cost_per_hr": 4.5})
    runner.main()
    assert runner.T0 == 1000.0 and runner.over_wall()  # a preempted run does not get a fresh wall clock
    st = json.loads((tmp_path / "runs" / "state.json").read_text())
    assert st["restarts"] == 1 and st["t0"] == 1000.0


def test_runner_on_a_pod_is_unchanged(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "state.json").write_text(json.dumps({"t0": 1000.0, "restarts": 0, "log": []}))
    runner = _runner(monkeypatch, tmp_path, {**CFG, "ack_wait_min": 0})
    calls = []
    monkeypatch.setattr(runner, "own_cost_per_hr", lambda: calls.append(1) or 3.0)
    monkeypatch.setattr(runner, "self_end", lambda acked: calls.append("end"))
    assert runner.main() is None
    assert calls == [1, "end"] and runner.WALL_H == pytest.approx(3.0)  # rate from RunPod; ack wait, self-end
    assert runner.T0 != 1000.0  # the pod's clock starts at this process
    assert runner.HARNESS == tmp_path / "code"  # HARNESS_DIR set by the test; the pod default is W/boot/code


def test_runner_harness_default_is_the_pod_path(tmp_path, monkeypatch):
    runner = _runner(monkeypatch, tmp_path, CFG)
    monkeypatch.delenv("HARNESS_DIR")
    sys.modules.pop("runner", None)
    runner = importlib.import_module("runner")
    assert runner.HARNESS == tmp_path / "boot" / "code"


def test_fetch_pulls_the_volume_into_the_pod_layout_and_ends_after_the_final_phase(tmp_path, monkeypatch):
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("ftest")
    vol = tmp_path / "vol" / "ftest" / "runs"
    (vol / "honeypot").mkdir(parents=True)
    (vol / "honeypot" / "manifest.json").write_text("{}")
    (vol / "state.json").write_text(json.dumps({"phase": "playing", "progress": {"branches_done": 3}}))
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if "modal" not in cmd:  # pod.place's rm -rf (the patch is on the shared subprocess module)
            return real_run(cmd, **kw)
        assert cmd[-3:-1] == ["collusion-runs", "ftest/runs"] and "--force" in cmd
        shutil.copytree(vol, Path(cmd[-1]) / "runs")  # `modal volume get` puts the folder under the destination
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(modal_pod.subprocess, "run", fake_run)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: True)
    modal_pod.save({"app_id": "ap-x", "created": time.time(), "cost_per_hr": 4.5, "budget_usd": 20,
                    "wall_hours": 8, "client_pid": 1})
    r = modal_pod.fetch_once()
    assert r["phase"] == "playing" and r["progress"] == {"branches_done": 3} and not r.get("ended")
    assert (tmp_path / "runs" / "ftest" / "manifest.json").exists()
    (vol / "state.json").write_text(json.dumps({"phase": "done"}))
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: False)  # the `modal run` client returned
    assert not modal_pod.fetch_once().get("ended")  # this download may span the end: one more full download first
    assert modal_pod.fetch_once().get("ended")


def _fake_modal(monkeypatch, vol, stop_ok=True, get_ok=True):
    """subprocess.run for `modal volume get` (copies `vol` like the CLI) and `modal app stop`; other calls pass."""
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    real_run, cmds = subprocess.run, []

    def fake_run(cmd, **kw):
        if "modal" not in cmd:
            return real_run(cmd, **kw)
        cmds.append(cmd)
        if "volume" in cmd:
            if get_ok and vol.exists():
                shutil.copytree(vol, Path(cmd[-1]) / "runs")
            return subprocess.CompletedProcess(cmd, 0 if get_ok else 1, "", "" if get_ok else "net down")
        return subprocess.CompletedProcess(cmd, 0 if stop_ok else 1, "", "")
    monkeypatch.setattr(modal_pod.subprocess, "run", fake_run)
    return cmds


def _setup(tmp_path, monkeypatch, name, state=None, **rec):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out(name)
    vol = tmp_path / "vol" / name / "runs"
    (vol / "honeypot").mkdir(parents=True)
    if state is not None:
        (vol / "state.json").write_text(json.dumps(state))
    modal_pod.save({"app_id": "ap-y", "created": time.time(), "cost_per_hr": 4.5, "budget_usd": 10,
                    "wall_hours": 8, "client_pid": 1, **rec})
    return vol


def test_fetch_stops_the_app_over_budget_counting_from_the_container_start(tmp_path, monkeypatch):
    vol = _setup(tmp_path, monkeypatch, "btest", {"phase": "playing", "t0": time.time() - 3 * 3600},
                 created=time.time() - 4 * 3600)
    cmds = _fake_modal(monkeypatch, vol)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: True)
    r = modal_pod.fetch_once()
    assert r["stop_confirmed"] and r["stop_reason"] == "budget" and not r.get("ended")
    assert any(c[-3:] == ["stop", "ap-y", "-y"] for c in cmds)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: False)  # stopped: the client returns
    assert not modal_pod.fetch_once().get("ended")
    assert modal_pod.fetch_once().get("ended")  # only after a full fetch that began after the end


def test_queue_time_does_not_count_as_run_time(tmp_path, monkeypatch):
    vol = _setup(tmp_path, monkeypatch, "qtest", {"phase": "playing", "t0": time.time() - 600},
                 created=time.time() - 1.5 * 3600, budget_usd=5)  # 1.5 h queued, then 10 min running
    cmds = _fake_modal(monkeypatch, vol)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: True)
    r = modal_pod.fetch_once()
    assert not r.get("stop_confirmed") and r["phase"] == "playing" and modal_pod.accrued(r) < 1.0
    assert not any("stop" in c for c in cmds)


def test_no_container_after_the_queue_limit_stops_the_app(tmp_path, monkeypatch):
    vol = _setup(tmp_path, monkeypatch, "q2test", None, created=time.time() - 2.5 * 3600)
    _fake_modal(monkeypatch, vol)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: True)
    r = modal_pod.fetch_once()
    assert r["stop_reason"] == "no container after the queue limit" and r["stop_confirmed"]


def test_a_failed_final_download_keeps_fetching(tmp_path, monkeypatch):
    vol = _setup(tmp_path, monkeypatch, "dtest", {"phase": "done", "t0": time.time() - 600})
    _fake_modal(monkeypatch, vol, get_ok=False)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: False)
    assert not modal_pod.fetch_once().get("ended")
    _fake_modal(monkeypatch, vol, get_ok=True)
    assert not modal_pod.fetch_once().get("ended")
    assert modal_pod.fetch_once().get("ended")


def test_down_ends_nothing_until_the_stop_is_confirmed_and_synced(tmp_path, monkeypatch):
    vol = _setup(tmp_path, monkeypatch, "ntest", {"phase": "playing", "t0": time.time() - 600})
    _fake_modal(monkeypatch, vol, stop_ok=False)
    modal_pod.down(None)
    r = modal_pod.load()
    assert not r.get("ended") and not r.get("stop_confirmed")  # still monitored, breaker still armed
    _fake_modal(monkeypatch, vol, stop_ok=True)
    modal_pod.down(None)
    assert modal_pod.load()["stop_confirmed"] and not modal_pod.load().get("ended")
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: False)
    assert not modal_pod.fetch_once().get("ended")
    assert modal_pod.fetch_once().get("ended")


def test_a_late_app_id_is_picked_up(tmp_path, monkeypatch):
    vol = _setup(tmp_path, monkeypatch, "ltest", None, app_id=None)
    _fake_modal(monkeypatch, vol)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: True)
    md = modal_pod.meta_dir()
    md.mkdir(parents=True, exist_ok=True)
    (md / "app.json").write_text(json.dumps({"app_id": "ap-late", "gpu": "H100"}))
    assert modal_pod.fetch_once()["app_id"] == "ap-late"


def test_a_launch_that_never_got_an_app_ends_once_the_client_is_gone(tmp_path, monkeypatch):
    vol = _setup(tmp_path, monkeypatch, "ftest2", None, app_id=None)
    _fake_modal(monkeypatch, vol)
    monkeypatch.setattr(modal_pod, "_alive", lambda pid: False)
    assert modal_pod.fetch_once()["phase"].startswith("launch failed")


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pod, "MAIN", tmp_path)
    pod.set_out("drytest")
    modal_pod.launch(_args("--dry-run"))
    assert "-m modal run --detach" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()
