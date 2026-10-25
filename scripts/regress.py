"""Regression check for the 22 Oct restructure: the same inputs must give the same outputs before and after.

  python scripts/regress.py run --impl new --root . --out runs/regress-new
  python scripts/regress.py compare runs/regress-ref runs/regress-new     # against a reference run (e.g. the old
                                                                          # harness: --impl old, before 22 Oct)

Cases:
  fake-*   deterministic fake models: self-play (2/3 firms x chat/no chat) and a honeypot run with a fake firm C
  robots   the known-answer robot check (144 cases)
  replay-* real recorded games (Luna, Haiku: self-play + honeypot) replayed: every prompt is looked up by its exact
           text in the recording and answered with the recorded reply. A prompt that is not in the recording is a
           MISS (the code changed what a model sees). Concurrency 1, so the order of identical prompts is fixed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import re
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path.home() / "frontier/apart-collusion-sprint"  # the old repo (impl "old": harness/bertrand)
GOLDEN = ROOT / "runs/golden"  # recorded real games (git-ignored; copied from the old repo's runs/ on 22 Oct)
RECORDINGS = {  # name -> (kind, recorded run dir)
    "replay-luna-selfplay": ("selfplay", GOLDEN / "smoke-v2-luna/f3-chat"),
    "replay-haiku-selfplay": ("selfplay", GOLDEN / "smoke-v2-haiku/f3-chat"),
    "replay-luna-honeypot": ("trap", GOLDEN / "trap-v2-luna"),
    "replay-haiku-honeypot": ("trap", GOLDEN / "trap-v2-claude-haiku"),
}
VOLATILE = {"ts", "created", "started", "finished", "latency_s", "git", "argv", "serving", "elapsed_s", "wall_s"}


def impl(name: str, root: Path | None):
    """The functions of one implementation: LLM module, selfplay, selfplay_stats, run, analyze, render,
    robot_check."""
    if name == "old":
        sys.path.insert(0, str(ARCHIVE / "harness"))
        llm = importlib.import_module("bertrand.llm")
        t = importlib.import_module("bertrand.trap")
        t.RESUME_WAIT_S = 0.0  # a replay miss is not a network error: fail fast instead of waiting 4 x 60 s
        return llm, t.selfplay, t.selfplay_stats, t.run, t.analyze, t.render, t.robot_check
    sys.path.insert(0, str(root))
    llm = importlib.import_module("collusion.clients.llm")
    sp = importlib.import_module("collusion.experiments.selfplay")
    hp = importlib.import_module("collusion.experiments.honeypot")
    hp.RESUME_WAIT_S = 0.0  # fail fast on a replay miss
    sc = importlib.import_module("collusion.analysis.scoring")
    rp = importlib.import_module("collusion.analysis.report")
    rb = importlib.import_module("collusion.experiments.robots")
    return llm, sp.selfplay, sc.selfplay_stats, hp.run, sc.analyze, rp.render, rb.robot_check


def _key(game: str, model: str, messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps([game, model, messages], sort_keys=True).encode()).hexdigest()


def _is_game(v) -> bool:
    return hasattr(v, "out") and hasattr(v, "rounds") and hasattr(v, "firms")


def calling_game():
    """The Game whose firm (or firm C) is asking: found in the callers' locals or in the closures of their
    callbacks (log_call / log lambdas). Identical prompts occur in a trunk and in all copies of a save point, so a
    reply must come from the same game's recording."""
    f = inspect.currentframe()
    while f:
        for v in list(f.f_locals.values()):
            cands = [v, *(v.values() if isinstance(v, dict) else [])]
            for c in cands:
                if _is_game(c):
                    return c
                for cell in getattr(c, "__closure__", None) or ():
                    try:
                        if _is_game(cell.cell_contents):
                            return cell.cell_contents
                    except ValueError:
                        pass
        f = f.f_back
    raise RuntimeError("replay: no calling game found")


def replay_llm(llm_mod, rec_dir: Path, log_path: Path, out_dir: Path):
    """An LLM whose chat() answers from a recording (calls.jsonl, plant.jsonl and confederate.jsonl)."""
    book = defaultdict(deque)
    for name in ("calls.jsonl", "confederate.jsonl", "plant.jsonl"):
        for f in sorted(rec_dir.rglob(name)):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("messages") is None or r.get("model") is None:
                    continue
                book[_key(str(f.parent.relative_to(rec_dir)), r["model"], r["messages"])].append(r)

    class Replay(llm_mod.LLM):
        misses = 0
        hits = 0

        async def chat(self, model, messages, temperature, max_tokens, system_as_user=False, top_p=None,
                       reasoning_effort=None, reasoning_summary=None, chat_template_kwargs=None,
                       prompt_cache_key=None):
            game = str(Path(calling_game().out).resolve().relative_to(out_dir.resolve()))
            k = _key(game, model, messages)
            with log_path.open("a") as fh:  # what was asked, minus the prompt text (compared old vs new)
                fh.write(json.dumps({"game": game, "model": model, "prompt_sha": k, "temperature": temperature,
                                     "max_tokens": max_tokens, "system_as_user": system_as_user, "top_p": top_p,
                                     "reasoning_effort": reasoning_effort, "reasoning_summary": reasoning_summary,
                                     "chat_template_kwargs": chat_template_kwargs,
                                     "prompt_cache_key": prompt_cache_key, "hit": bool(book.get(k))}) + "\n")
            if not book.get(k):
                Replay.misses += 1
                raise LookupError(f"replay miss: {model} prompt {k[:12]} is not in the recording")
            Replay.hits += 1
            r = book[k].popleft()
            book[k].append(r)  # identical prompts recorded more often than replayed: cycle in recorded order
            return llm_mod.Reply(content=r.get("content") or "", reasoning=r.get("reasoning"),
                                 usage=r.get("usage") or {}, finish_reason=r.get("finish_reason"), latency_s=0.0)

    return Replay()


def logging_llm(llm_mod, log_path: Path, out_dir: Path):
    """The real (fake-model) LLM, but every request's parameters are logged like a replay's (no-chat top_p etc.)."""

    class Logged(llm_mod.LLM):
        async def chat(self, model, messages, temperature, max_tokens, system_as_user=False, top_p=None,
                       reasoning_effort=None, reasoning_summary=None, chat_template_kwargs=None,
                       prompt_cache_key=None):
            game = str(Path(calling_game().out).resolve().relative_to(out_dir.resolve()))
            with log_path.open("a") as fh:
                fh.write(json.dumps({"game": game, "model": model, "prompt_sha": _key(game, model, messages),
                                     "temperature": temperature, "max_tokens": max_tokens,
                                     "system_as_user": system_as_user, "top_p": top_p,
                                     "reasoning_effort": reasoning_effort, "reasoning_summary": reasoning_summary,
                                     "chat_template_kwargs": chat_template_kwargs,
                                     "prompt_cache_key": prompt_cache_key}) + "\n")
            return await super().chat(model, messages, temperature, max_tokens, system_as_user, top_p,
                                      reasoning_effort, reasoning_summary, chat_template_kwargs, prompt_cache_key)

    return Logged()


async def run_cases(which: str, root: Path | None, out: Path, only: list[str] | None):
    llm_mod, selfplay, selfplay_stats, run, analyze, render, robot_check = impl(which, root)

    def scored_selfplay(d: Path):
        (d / "stats.json").write_text(json.dumps(selfplay_stats(d), indent=2, sort_keys=True))

    def scored_run(d: Path):
        (d / "render.md").write_text(render(analyze(d)))
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    def want(c):
        return not only or c in only

    for firms in (2, 3):
        for chat in (False, True):
            c = f"fake-selfplay-f{firms}-{'chat' if chat else 'nochat'}"
            if want(c):
                await selfplay("fake:1.6:Steady here" if chat else "fake:1.6", out / c, games=2, periods=12,
                               seed0=7, chat=chat, firms=firms, concurrency=1,
                               sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
                               llm=logging_llm(llm_mod, out / f"{c}.asked.jsonl", out / c))
                scored_selfplay(out / c)
                report[c] = "ok"
    c = "fake-honeypot"
    if want(c):
        await run("fake:1.65:Steady here", out / c, games=2, ks=[5, 8], copy_len=14, seed0=3,
                  c_model="fake-raw:My message: Demand held up fine this round.", c_effort=None, concurrency=1,
                  sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
                  llm=logging_llm(llm_mod, out / f"{c}.asked.jsonl", out / c))
        scored_run(out / c)
        report[c] = "ok"
    c = "robots"
    if want(c):
        res = await robot_check(out / c)
        report[c] = {k: res[k] for k in res if k in ("pass", "cases")} or "done"
    for c, (kind, rec) in RECORDINGS.items():
        if not want(c):
            continue
        man = json.loads((rec / "manifest.json").read_text())
        llm = replay_llm(llm_mod, rec, out / f"{c}.asked.jsonl", out / c)
        if kind == "selfplay":
            await selfplay(man["model"], out / c, games=man["games"], periods=man["periods"], seed0=man["seed0"],
                           sampling=man["sampling"], chat=man["chat"], firms=man.get("firms", 3), concurrency=1,
                           llm=llm)
            scored_selfplay(out / c)
        else:
            await run(man["model"], out / c, games=man["games"], ks=man["ks"], copy_len=man["copy_len"],
                      reps=man["reps"], steps=man["steps"], seed0=man["seed0"], c_model=man["c_model"],
                      c_effort=man["c_effort"], sampling=man["sampling"], concurrency=1, llm=llm,
                      hold=man["constants"]["hold"])
            scored_run(out / c)
        report[c] = {"hits": type(llm).hits, "misses": type(llm).misses}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def _strip(text: str, out_dirs: tuple[str, ...]) -> str:
    for o in out_dirs:  # the run dir appears as given (relative) and resolved (absolute)
        text = text.replace(o + "/", "<OUT>/")
    # ...and relative to whatever directory the run was started from
    return re.sub(rf"[^\s\"']*{re.escape(Path(out_dirs[0]).name)}/", "<OUT>/", text)


def _clean(x, out_dirs: tuple[str, ...]):
    if isinstance(x, dict):
        return {k: _clean(v, out_dirs) for k, v in sorted(x.items()) if k not in VOLATILE}
    if isinstance(x, list):
        return [_clean(v, out_dirs) for v in x]
    if isinstance(x, str):
        return _strip(x, out_dirs)
    return x


def normalized(path: Path, out_dir: Path) -> list:
    text = path.read_text()
    o = (str(out_dir.resolve()), str(out_dir))
    if path.suffix == ".jsonl":
        return [_clean(json.loads(l), o) for l in text.splitlines() if l.strip()]
    if path.suffix == ".json":
        return [_clean(json.loads(text), o)]
    return [re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.+Z-]+", "<TS>", _strip(text, o))]


def compare(a: Path, b: Path, ignore_recorded_vs_replayed: bool = False) -> int:
    # .run.lock: the honeypot run folder's (empty, released) OS-lock file, no data (24 Oct)
    fa = {p.relative_to(a) for p in a.rglob("*") if p.is_file() and p.name != ".run.lock"}
    fb = {p.relative_to(b) for p in b.rglob("*") if p.is_file() and p.name != ".run.lock"}
    bad = 0
    for p in sorted(fa ^ fb):
        print(f"ONLY IN {'old' if p in fa else 'new'}: {p}")
        bad += 1
    for p in sorted(fa & fb):
        if p.suffix not in (".json", ".jsonl", ".md", ".txt", ".log"):
            continue
        if normalized(a / p, a) != normalized(b / p, b):
            print(f"DIFFERS: {p}")
            bad += 1
    print(f"{len(fa & fb)} common files, {bad} problems")
    return bad


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--impl", choices=["old", "new"], required=True)
    r.add_argument("--root", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--only", nargs="*")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    a = ap.parse_args()
    if a.cmd == "run":
        if a.out.exists():
            shutil.rmtree(a.out)
        asyncio.run(run_cases(a.impl, a.root, a.out, a.only))
    else:
        sys.exit(1 if compare(a.a, a.b) else 0)


if __name__ == "__main__":
    main()
