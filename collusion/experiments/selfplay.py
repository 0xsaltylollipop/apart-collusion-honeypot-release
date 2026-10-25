"""Answer key: self-play with no planted firm (2 or 3 copies of the tested model, chat or no chat)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from collusion.clients.llm import LLM
from collusion.experiments.common import _finished
from collusion.experiments.honeypot import MARKET
from collusion.game.engine import Game, make_config, report_failures, restore
from collusion.game.prompts import PROMPT_VERSION
from collusion.market.economics import PRESETS


def selfplay_market(firms: int = 3):
    """3 firms: the trap's market (LLMCartel triopoly). 2 firms: the Calvano/Fish duopoly (Nash 1.47, monopoly 1.93)."""
    if firms not in (2, 3):
        raise ValueError("self-play supports 2 or 3 firms")
    return MARKET if firms == 3 else PRESETS["duopoly"]


def selfplay_config(model: str, seed: int, periods: int, sampling: dict, *, chat: bool = False, window: int = 100,
                    firms: int = 3, tag: str = "pre-sprint-selfplay") -> dict:
    """`firms` copies of the tested model, no firm C. chat=False (the private-line design's answer key): the trap
    subjects' prompt without the messaging block (P0 wording, strict parser, 3 tries)."""
    m = selfplay_market(firms)
    return make_config(
        market={"a": list(m.a), "c": list(m.c), "mu": m.mu, "a0": m.a0, "alpha": m.alpha, "beta": m.beta},
        firms=[{"kind": "llm", "model": model} for _ in range(firms)], prompt_mode="chat" if chat else "p0",
        chat_delivery=chat, parse_mode="strict", retries=3, periods=periods, window=window, seed=seed, tag=tag,
        label=f"sp-s{seed}", run_mode="selfplay", prompt_cache=True, **sampling)


async def selfplay(model: str, out: Path, games: int = 3, periods: int = 300, seed0: int = 1,
                   sampling: dict | None = None, chat: bool = False, concurrency: int = 16, llm: LLM | None = None,
                   firms: int = 3, tag: str = "pre-sprint-selfplay") -> dict:
    """Plays (or resumes) `games` self-play games of `periods` rounds under out/sp-s<seed>."""
    llm = llm or LLM()
    sampling = sampling or {}
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = {"mode": "selfplay", "model": model, "games": games, "periods": periods, "seed0": seed0, "chat": chat,
           "sampling": sampling}
    if firms != 3:  # older 3-firm manifests have no "firms" key
        man["firms"] = firms
    if chat and firms > 2:  # only 3-firm chat changed in prompt v2; a v1 dir is refused, never mixed with v2 games
        man["prompt_version"] = PROMPT_VERSION
    if mpath.exists():
        old = json.loads(mpath.read_text())
        changed = {kk: (old.get(kk), man.get(kk)) for kk in set(man) | set(old) - {"finished"}
                   if old.get(kk) != man.get(kk)}
        if changed:
            raise SystemExit(f"{out} was run with a different design {changed}; use a new directory")
    mpath.write_text(json.dumps(man, indent=2))
    todo = []
    for g in range(games):
        d = out / f"sp-s{seed0 + g}"
        if _finished(d, periods):
            continue
        if (d / "metadata.json").exists() and (d / "rounds.jsonl").exists():
            todo.append(restore(d, None, d, llm=llm))
        else:
            cfg = selfplay_config(model, seed0 + g, periods, sampling, chat=chat, firms=firms, tag=tag)
            todo.append(Game(cfg, d, llm=llm, overwrite=True))
    sem = asyncio.Semaphore(concurrency)

    async def one(game):
        async with sem:
            for attempt in range(4):  # a crashed game resumes itself instead of idling until its siblings finish
                try:
                    return await game.run(periods)
                except Exception as e:
                    if attempt == 3:
                        raise
                    print(f"{game.out.name}: {e!r}; resuming in 60 s (attempt {attempt + 2})", flush=True)
                    await asyncio.sleep(60)
                    game = restore(game.out, None, game.out, llm=llm)

    res = await asyncio.gather(*(one(g) for g in todo), return_exceptions=True)
    if report_failures(todo, res):
        raise SystemExit("self-play game failures; rerun to resume")
    return man
