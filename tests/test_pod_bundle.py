"""The pod's code bundle (scripts/pod/pod.py bundle_b64): small enough for RunPod's pod env, without the laptop-only
modules, and the shipped package runs the pod's entry points (honeypot run, self-play, analysis) on its own.
24 Oct: the gz bundle had grown to ~121 KB of base64 and every pod launch failed with HTTP 500."""

import ast
import base64
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pod"))
import pod  # noqa: E402

MARGIN_LIMIT = 80_000  # a tripwire 5 KB below the hard limit (24 Oct evening: ~65 KB with docstrings stripped), so growth shows up here before a launch fails


def _extract(tmp_path):
    b64 = pod.bundle_b64()
    print(f"pod code bundle: {len(b64)} bytes of base64 (test limit {MARGIN_LIMIT}, hard limit "
          f"{pod.BUNDLE_LIMIT_B64})")
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64)), mode="r:*") as tf:  # as the pod unpacks it
        tf.extractall(tmp_path)  # noqa: S202 (our own archive)
    return b64, tmp_path / "code"


def _cli(code, *args):
    return subprocess.run([sys.executable, "-m", "collusion", *args], cwd=code, capture_output=True, text=True,
                          timeout=300)


def test_bundle_is_small_and_ships_only_what_the_pod_needs(tmp_path):
    b64, code = _extract(tmp_path)
    assert len(b64) < MARGIN_LIMIT <= pod.BUNDLE_LIMIT_B64 < 91_000  # 91 KB was the last size RunPod accepted
    shipped = {p.relative_to(code).as_posix() for p in code.rglob("*.py")}
    assert not shipped & pod.LAPTOP_ONLY and "__pycache__" not in str(shipped)
    assert {"collusion/cli.py", "collusion/analysis/scoring.py", "collusion/experiments/honeypot.py"} <= shipped
    assert (tmp_path / "runner.py").exists() and (tmp_path / "selfstop.py").exists()
    data = {p.relative_to(ROOT).as_posix() for p in (ROOT / "collusion").rglob("*")
            if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts}
    assert data and all((code / rel).read_bytes() == (ROOT / rel).read_bytes() for rel in data)  # e.g. prices.json
    for rel in shipped:  # comments and docstrings stripped, code and line numbers unchanged
        orig, bare = (ROOT / rel).read_text(), (code / rel).read_text()
        assert ast.dump(pod._without_docstrings(ast.parse(orig), pod.strip_comments(orig))) == ast.dump(
            ast.parse(bare)), rel
        assert orig.count("\n") == bare.count("\n"), rel


def test_the_shipped_package_runs_the_pod_entry_points(tmp_path):
    _, code = _extract(tmp_path / "boot")
    where = subprocess.run([sys.executable, "-c", "import collusion; print(collusion.__file__)"], cwd=code,
                           capture_output=True, text=True)
    assert Path(where.stdout.strip()).resolve().is_relative_to(code.resolve())  # the bundle, not this checkout
    hp_out, sp_out = tmp_path / "hp", tmp_path / "sp"
    r = _cli(code, "run", "--model", "fake:1.6:Steady here", "--out", str(hp_out), "--games", "2", "--ks", "10",
             "--copy-len", "4", "--hold", "2", "--c-model", "fake-raw:My message: Our demand held up fine.",
             "--c-effort", "", "--sub-ks", "5", "--sub-games", "1", "--seed0", "11")
    assert r.returncode == 0, r.stderr[-2000:]
    man = json.loads((hp_out / "manifest.json").read_text())
    assert sorted(man["units"]) == ["trunk-s11@k10", "trunk-s11@k5", "trunk-s12@k10"]
    assert _cli(code, "analyze", str(hp_out)).returncode == 0
    r = _cli(code, "selfplay", "--model", "fake:1.6:Steady here", "--out", str(sp_out), "--games", "1",
             "--periods", "5", "--chat")
    assert r.returncode == 0, r.stderr[-2000:]
    assert _cli(code, "selfplay-stats", str(sp_out)).returncode == 0
    gone = _cli(code, "crossmodel", "--pairs", "p.json", "--out", str(tmp_path / "cm"))
    assert gone.returncode == 2 and "invalid choice" in gone.stderr  # laptop-only, not on the pod


def test_the_k5_schedule_is_checked_on_the_laptop_before_launch():
    ok = {"games": 10, "ks": [10], "sub_ks": [5], "sub_games": 3}
    pod.check_sub_schedule(ok)
    pod.check_sub_schedule({"games": 10, "ks": [10]})
    for bad in ({"sub_ks": [5]}, {"sub_games": 3}, {"sub_ks": [5, 5], "sub_games": 3}, {"sub_ks": [10], "sub_games": 3},
                {"sub_ks": [12], "sub_games": 3}, {"sub_ks": ["5"], "sub_games": 3}, {"sub_ks": [5], "sub_games": 11},
                {"sub_ks": [5], "sub_games": 3.0}, {"sub_ks": 5, "sub_games": 3}, {"sub_ks": [0], "sub_games": 3}):
        with pytest.raises(SystemExit):
            pod.check_sub_schedule({"games": 10, "ks": [10], **bad})


def test_strip_comments_keeps_every_line_even_with_form_feeds():
    src = "x = 1  # one\n\x0cy = 2\ns = '# not a comment'\n"
    bare = pod.strip_comments(src)
    assert bare == "x = 1\n\x0cy = 2\ns = '# not a comment'\n"
    assert ast.dump(ast.parse(bare)) == ast.dump(ast.parse(src))


def test_docstring_stripping_keeps_lines_and_code():
    q = '"' * 3
    src = "\n".join([
        f"{q}Module doc", f"with two lines.{q}", "import os", "", "",
        "def f(x):  # c", f"    {q}Doc \u00e9.{q}", "    return x  # tail", "", "",
        "class A:", f"    {q}Multi", "    line", f"    doc.{q}", f"    def g(self): {q}one{q}; return 1", "", "",
        "def h():", f"    {q}only a doc{q}",
        "def k():", f"    {q}multi", f"    line{q}; y = 2", "    return y",
        "def n():", f"    {q}multi", f"    then a comment{q}  # c", "    return 5",
        "def m():", f"    {q}at the end", f"    of the file{q}"])  # no final newline
    bare = pod.strip_docstrings(pod.strip_comments(src))
    assert bare.count("\n") == src.count("\n") and "only a doc" not in bare
    assert ast.dump(ast.parse(bare)) == ast.dump(pod._without_docstrings(ast.parse(src), pod.strip_comments(src)))
    ns = {}
    exec(compile(bare, "x", "exec"), ns)
    assert ns["f"](3) == 3 and ns["A"]().g() == 1 and ns["h"]() is None and ns["k"]() == 2 and ns["m"]() is None and ns["n"]() == 5
