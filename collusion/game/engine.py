"""Game loop, per-period state saving, resume and fork.

Layout of a game directory:
  metadata.json   config, benchmarks, git commit, argv, parent (for forks)
  rounds.jsonl    one line per period (0 = the initial prior round); each line is
                  enough, together with metadata.json, to resume from that period
  calls.jsonl     one line per LLM call attempt
  summary.json    written at the end (also after a crash): cost, tokens, parse failures, truncations, error

A run or fork never writes into an existing game directory unless overwrite=True (in-place resume
continues the same directory on purpose).

Interventions (list of dicts in config["interventions"]):
  {"type": "inject_message", "period": k, "to": "all"|[j..], "from": i, "text": "..."}
      the text appears as firm i's message in round k's market data of each recipient,
      so recipients see it when deciding period k+1
  {"type": "force_price", "period": k, "firm": i, "price": 1.47 | "nash"}
      firm i's price in period k is set without asking the model (punishment test)
  {"type": "chat_delivery", "from_period": k, "on": true|false}
      note: inject_message bypasses chat_delivery (an injected message is delivered even when delivery
      is off), so a trap can be planted into a "no delivery" game on purpose
  {"type": "notice", "period": k, "to": "all"|[j..], "text": "..."}
      an exchange notice shown in round k's market data of each recipient (seen when deciding period
      k+1) as "Exchange notice: ..."; used for the honeypot test-cue arm
  {"type": "set_scripted", "from_period": k, "firm": i, "spec": {...}}
      from period k on, scripted firm i behaves as its config spec updated with `spec`
      (e.g. a plant switching from {"price_rule": "nash"} to {"price_rule": "fixed", "price": 1.8});
      switches are re-applied on resume, so a resumed branch keeps the switched behaviour.
      A scripted firm whose spec has "llm_firm": {"model": ...} is LLM-driven (price and message from an LLM with
      the LLM firms' own prompt, calls in plant.jsonl); a switch with "llm_firm": null hands it back to its
      scripted rule (and to a "confederate" spec, if any, for its words).
  {"type": "set_persona", "from_period": k, "firm": i, "text": "..."}
      from period k on, LLM firm i gets `text` added to its system prompt (honeypot positive control: the colluder
      instruction from the round after the save point only, so instructed copies share the plain trunk; 24 Oct)
Interventions are validated up front (known type, integer firm indices in range, periods inside the
game and, for forks, after the fork point); a bad one raises instead of being silently ignored.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from collusion.clients.costs import call_cost
from collusion.market.economics import Market
from collusion.game import plant as confederate
from collusion.game.prompts import PROMPT_VERSION
from collusion.game.firms import Decision, LLMFirm, ScriptedFirm, make_firms
from collusion.clients.llm import LLM, route

DEFAULTS = {
    "prompt_mode": "p0",
    "chat_delivery": True,
    "private_to": None,   # chat: firm index; LLM firms then message only that firm (private line), see prompts.build_chat
    "periods": 300,
    "window": 100,
    "ceiling_mult": 1.5,
    "temperature": 1.0,
    "max_tokens": 4096,
    "top_p": None,        # p0: 1.0 (LLMCartel's vLLM SamplingParams); else provider default
    "context_cap": None,  # p0: 8192 -> max_tokens = min(max_tokens, 8192 - prompt_tokens) on vLLM, as LLMCartel
    "parse_mode": None,   # p0: "lcartel" (their parser); else "strict"
    "retries": None,      # lcartel: 9 attempts (their first try + 8 retries); strict: 3
    "system_as_user": False,
    "reasoning_effort": None,   # OpenAI reasoning models: set -> Responses API with this effort (LLMCartel GPT-5.4: medium)
    "reasoning_summary": None,  # Responses API reasoning summary (LLMCartel: auto)
    "chat_template_kwargs": None,  # vLLM only, e.g. {"enable_thinking": true}
    "seed": 0,
    "tag": "pre-sprint",
    "label": "",
    "interventions": [],
}


def make_config(**kw) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update({k: v for k, v in kw.items() if v is not None or k not in DEFAULTS})
    return resolve_config(cfg)


def resolve_config(cfg: dict) -> dict:
    """Fill mode-dependent defaults (LLMCartel reproduction settings for p0)."""
    p0 = cfg.get("prompt_mode") == "p0"
    if cfg.get("parse_mode") is None:
        cfg["parse_mode"] = "lcartel" if p0 else "strict"
    if cfg.get("retries") is None:
        # lcartel: 9 attempts (r1_duopoly_tester.py); lcartel-gpt: 3 (chatgpt_duopoly_tester.py, 1 + 2 retries)
        cfg["retries"] = 9 if cfg["parse_mode"] == "lcartel" else 3
    if cfg.get("top_p") is None and p0:
        cfg["top_p"] = 1.0
    if cfg.get("context_cap") is None and p0:
        cfg["context_cap"] = 8192
    if cfg.get("context_cap") == 0:
        cfg["context_cap"] = None
    return cfg


INTERVENTION_TYPES = {"inject_message", "notice", "force_price", "chat_delivery", "set_scripted", "set_persona"}


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_interventions(ivs: list[dict], n_firms: int, periods: int, fork_at: int | None = None):
    """Raise ValueError on any intervention that would otherwise be silently ignored or misapplied."""
    def bad(iv, why):
        raise ValueError(f"invalid intervention {iv!r}: {why}")

    def firm_ok(j):
        return _is_int(j) and 0 <= j < n_firms

    for iv in ivs:
        if not isinstance(iv, dict) or iv.get("type") not in INTERVENTION_TYPES:
            bad(iv, f"type must be one of {sorted(INTERVENTION_TYPES)}")
        t = iv["type"]
        key = "from_period" if t in ("chat_delivery", "set_scripted", "set_persona") else "period"
        k = iv.get(key)
        if not _is_int(k):
            bad(iv, f"{key} must be an integer")
        if not (0 <= k <= periods):
            bad(iv, f"{key} {k} outside the game (0..{periods})")
        if t in ("inject_message", "notice"):
            if t == "inject_message" and not firm_ok(iv.get("from")):
                bad(iv, f"'from' must be an integer firm index in 0..{n_firms - 1}")
            to = iv.get("to", "all")
            if to != "all" and not (isinstance(to, list) and to and all(firm_ok(j) for j in to)):
                bad(iv, "'to' must be \"all\" or a non-empty list of integer firm indices")
            if not isinstance(iv.get("text"), str) or not iv["text"].strip():
                bad(iv, "'text' must be a non-empty string")
            if fork_at is not None and k < fork_at:
                bad(iv, f"period {k} is before the fork point {fork_at} (would rewrite shared history)")
        else:
            if k < 1:
                bad(iv, f"{key} must be >= 1")
            if fork_at is not None and k <= fork_at:
                bad(iv, f"{key} {k} must be after the fork point {fork_at}")
            if t == "force_price":
                if not firm_ok(iv.get("firm")):
                    bad(iv, f"'firm' must be an integer firm index in 0..{n_firms - 1}")
                pr = iv.get("price")
                if not (pr == "nash" or (isinstance(pr, (int, float)) and not isinstance(pr, bool) and pr > 0)):
                    bad(iv, "'price' must be \"nash\" or a positive number")
            elif t == "set_persona":
                if not firm_ok(iv.get("firm")):
                    bad(iv, f"'firm' must be an integer firm index in 0..{n_firms - 1}")
                if not isinstance(iv.get("text"), str) or not iv["text"].strip():
                    bad(iv, "'text' must be a non-empty string")
            elif t == "set_scripted":
                if not firm_ok(iv.get("firm")):
                    bad(iv, f"'firm' must be an integer firm index in 0..{n_firms - 1}")
                if not isinstance(iv.get("spec"), dict) or not iv["spec"]:
                    bad(iv, "'spec' must be a non-empty dict of scripted-firm settings")
                if "kind" in iv["spec"] or "rule" in iv["spec"]:
                    bad(iv, "'spec' may not change the firm's kind or rule (only its settings)")
            elif not isinstance(iv.get("on"), bool):
                bad(iv, "'on' must be true or false")


def market_of(cfg: dict) -> Market:
    m = cfg["market"]
    return Market(a=tuple(m["a"]), c=tuple(m["c"]), mu=m.get("mu", 0.25), a0=m.get("a0", 0.0),
                  alpha=m.get("alpha", 1.0), beta=m.get("beta", 100.0))


def ceiling_of(cfg: dict, market: Market) -> float:
    b = market.benchmarks()
    # LLMCartel shows 1.5 x its (pairwise) monopoly price; same value for the duopoly.
    pm = b.get("pairwise_monopoly_prices", b["monopoly_prices"])
    return cfg["ceiling_mult"] * max(pm)


def git_info() -> dict:
    if os.environ.get("BERTRAND_GIT_COMMIT"):  # set when the harness runs from a tarball (e.g. on a GPU pod)
        return {"commit": os.environ["BERTRAND_GIT_COMMIT"], "harness_dirty": os.environ.get("BERTRAND_GIT_DIRTY") == "1"}
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "collusion"], cwd=root,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "harness_dirty": dirty}
    except Exception:
        return {"commit": None, "harness_dirty": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rng_dump(rng: random.Random) -> list:
    v, state, g = rng.getstate()
    return [v, list(state), g]


def _rng_load(dump: list) -> random.Random:
    rng = random.Random()
    rng.setstate((dump[0], tuple(dump[1]), dump[2]))
    return rng


async def _gather_or_cancel(coros: list) -> list:
    """asyncio.gather of one round's calls, but when one raises (or the round is cancelled) the others are cancelled
    AND awaited before the error propagates, so an abandoned call can never finish later and log a reply that no
    round used (Astra 24 Oct: the debrief then continued from such a reply). The tasks are created in the same order
    as gather would, so the normal path is unchanged."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class Game:
    def __init__(self, cfg: dict, out_dir: Path, llm: LLM | None = None, parent: dict | None = None,
                 overwrite: bool = False):
        self.cfg = resolve_config(cfg)
        self.out = Path(out_dir)
        self.llm = llm or LLM()
        self.market = market_of(cfg)
        self.bench = self.market.benchmarks()
        self.ceiling = ceiling_of(cfg, self.market)
        self.cost = [self.market.alpha * c for c in self.market.c]
        self.firms = make_firms(cfg, self.llm)
        self.rounds: list[dict] = []
        self.rng = random.Random(cfg["seed"])
        self.chat_delivery = cfg["chat_delivery"]
        self.parent = parent
        self.overwrite = overwrite
        self.error: str | None = None
        validate_interventions(cfg["interventions"], len(self.firms), cfg["periods"])
        for iv in cfg["interventions"]:
            if iv["type"] == "set_scripted" and not isinstance(self.firms[iv["firm"]], ScriptedFirm):
                raise ValueError(f"invalid intervention {iv!r}: firm {iv['firm']} is not a scripted firm")
            if iv["type"] == "set_persona" and not isinstance(self.firms[iv["firm"]], LLMFirm):
                raise ValueError(f"invalid intervention {iv!r}: firm {iv['firm']} is not an LLM firm")

    def apply_scripted_switches(self, t: int):
        """Rebuild each scripted firm from its config spec plus every set_scripted switch with
        from_period <= t (in order), so the behaviour at period t never depends on the run history."""
        for i, spec in enumerate(self.cfg["firms"]):
            if spec["kind"] != "scripted":
                continue
            merged = copy.deepcopy(spec)
            for iv in self.cfg["interventions"]:
                if iv["type"] == "set_scripted" and iv["firm"] == i and iv["from_period"] <= t:
                    merged.update(copy.deepcopy(iv["spec"]))
            self.firms[i] = ScriptedFirm(i, merged)

    # ---------- setup / persistence ----------
    def _prepare_dir(self):
        """Refuse to mix two games in one directory (rounds.jsonl/calls.jsonl are append-only)."""
        # every per-game file, so an overwrite never leaves the old game's plant/confederate calls or post-hoc
        # answers behind (ask_posthoc skips a branch whose posthoc.json exists)
        existing = [n for n in ("metadata.json", "rounds.jsonl", "calls.jsonl", "summary.json", "plant.jsonl",
                                "confederate.jsonl", "posthoc.json") if (self.out / n).exists()]
        if existing and not self.overwrite:
            raise FileExistsError(f"{self.out} already holds a game ({', '.join(existing)}); "
                                  "use resume to continue it, or pass overwrite to replace it")
        for n in existing:
            (self.out / n).unlink()
        self.out.mkdir(parents=True, exist_ok=True)

    def _write_metadata(self):
        self._prepare_dir()
        meta = {
            "created": _now(), "config": self.cfg, "benchmarks": self.bench, "ceiling": self.ceiling,
            "git": git_info(), "argv": sys.argv, "parent": self.parent, "prompt_version": PROMPT_VERSION,
            "models": [f.get("model") for f in self.cfg["firms"] if f["kind"] == "llm"],
            "serving": self.llm.serving_info(),
        }
        (self.out / "metadata.json").write_text(json.dumps(meta, indent=2))

    def _append(self, name: str, rec: dict):
        with open(self.out / name, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _initial_round(self) -> dict:
        # LLMCartel draws the prior round as Nash + randint(0, 11)/100 - 0.05
        prices = [p + (self.rng.randint(0, 11) - 5) / 100.0 for p in self.bench["nash_prices"]]
        n = len(prices)
        return self._finalize(0, prices, [None] * n, [False] * n, [False] * n, [False] * n)

    def _finalize(self, t: int, prices, sent, forced, parse_failed, truncated) -> dict:
        n = len(prices)
        q = self.market.quantities(prices).tolist()
        pi = self.market.profits(prices).tolist()
        inbox = [dict() for _ in range(n)]
        if self.chat_delivery:
            for i, msg in enumerate(sent):
                if not msg:
                    continue
                f = self.firms[i]
                hub = self.cfg.get("private_to")
                if isinstance(f, ScriptedFirm):
                    recips = f.recipients(n)
                elif hub is not None and i != hub:
                    recips = [hub]
                else:
                    recips = [j for j in range(n) if j != i]
                for j in recips:
                    inbox[j][str(i)] = msg
        rnd = {"period": t, "prices": list(map(float, prices)), "quantities": q, "profits": pi, "sent": sent,
               "inbox": inbox, "forced": forced, "parse_failed": parse_failed, "truncated": truncated,
               "chat_delivery": self.chat_delivery,
               "injected": []}
        apply_injections(rnd, self.cfg["interventions"])
        rnd["rng_state"] = _rng_dump(self.rng)
        return rnd

    # ---------- main loop ----------
    async def run(self, until: int | None = None):
        until = until or self.cfg["periods"]
        if not self.rounds:
            self._write_metadata()
            first = self._initial_round()
            self.rounds.append(first)
            self._append("rounds.jsonl", first)
        try:
            while self.rounds[-1]["period"] < until:
                t = self.rounds[-1]["period"] + 1
                await self.step(t)
        except BaseException as e:  # keep cost/tokens of what did run; the game can be resumed in place
            self.error = repr(e)
            self._write_summary()
            raise
        self._write_summary()
        return self

    def _log_call(self, t: int, firm: int, rec: dict, file: str = "calls.jsonl"):
        model = rec.get("model", "")
        cost = call_cost(model, rec.get("usage")) if "usage" in rec else 0.0
        if cost is None and route(model) in ("openai", "anthropic") and model not in _WARNED_UNPRICED:
            _WARNED_UNPRICED.add(model)
            print(f"WARNING: no price for {model!r} in prices.json; its cost is NOT counted", file=sys.stderr)
        self._append(file, {"ts": _now(), "tag": self.cfg["tag"], "game": self.out.name, "period": t,
                            "firm": firm, **rec, "cost_usd": cost})

    async def step(self, t: int):
        self.apply_scripted_switches(t)
        for iv in self.cfg["interventions"]:
            if iv["type"] == "chat_delivery" and iv["from_period"] == t:
                self.chat_delivery = bool(iv["on"])
        forced = {iv["firm"]: iv["price"] for iv in self.cfg["interventions"]
                  if iv["type"] == "force_price" and iv["period"] == t}
        base_ctx = {"period": t, "history": self.rounds, "cost": self.cost, "ceiling": self.ceiling,
                    "nash": self.bench["nash_prices"], "market": self.market,
                    # one cache key per game seed (shared by its copies, whose history up to the fork is identical)
                    "cache_key": f"{self.cfg.get('tag')}|s{self.cfg['seed']}" if self.cfg.get("prompt_cache") else None}
        ctx = base_ctx

        n = len(self.firms)
        decisions = [None] * n
        llm_jobs, conf_jobs = [], []
        for i, f in enumerate(self.firms):  # scripted firms first, in index order, so RNG use is deterministic
            if i in forced:
                continue
            if isinstance(f, ScriptedFirm) and f.spec.get("llm_firm"):
                # an LLM-driven plant: price AND message come from an LLM playing a normal firm with the same
                # prompt as the LLM firms (honeypot trunks and control branches); logged to plant.jsonl.
                # With a "confederate" spec as well, only the PRICE comes from the LLM firm and the words from
                # the confederate speaker (honeypot "neutral" warm-up: real prices, neutral chat)
                drv = f.spec["llm_firm"]
                plant_cfg = {**self.cfg, **{kk: v for kk, v in drv.items() if kk != "model"}}
                pf = LLMFirm(i, {"kind": "llm", "model": drv["model"]}, plant_cfg, self.llm)
                plant_ctx = dict(base_ctx, log_call=lambda rec, i=i: self._log_call(t, i, rec, "plant.jsonl"))
                llm_jobs.append((i, pf.decide(plant_ctx)))
                conf = f.spec.get("confederate")
                if conf:
                    conf_jobs.append((i, confederate.generate(
                        self.llm, conf, self.rounds, i, self.cost[i], self.ceiling, self.cfg, opener=False,
                        log=lambda rec, i=i: self._log_call(t, i, rec, "confederate.jsonl"))))
            elif isinstance(f, ScriptedFirm):
                decisions[i] = f.decide(ctx, self.rng)  # the price always comes from the scripted rule
                conf = f.spec.get("confederate")
                if conf:  # an AI confederate writes this scripted firm's message (see confederate.py)
                    # the "trusted" firm C is told this round's price and its own state, so its words match
                    conf_jobs.append((i, confederate.generate(
                        self.llm, conf, self.rounds, i, self.cost[i], self.ceiling, self.cfg, opener=False,
                        log=lambda rec, i=i: self._log_call(t, i, rec, "confederate.jsonl"),
                        price=decisions[i].price, firm_spec=f.spec)))
            else:
                firm_ctx = dict(base_ctx, log_call=lambda rec, i=i: self._log_call(t, i, rec))
                persona = [iv["text"] for iv in self.cfg["interventions"]
                           if iv["type"] == "set_persona" and iv["firm"] == i and iv["from_period"] <= t]
                if persona:  # only rounds with a switch get the key, so other games' calls are unchanged
                    firm_ctx["persona"] = persona[-1]
                llm_jobs.append((i, f.decide(firm_ctx)))
        jobs = llm_jobs + conf_jobs
        results = await _gather_or_cancel([job for _, job in jobs])
        conf_info = {}
        for (i, _), res in zip(llm_jobs, results[:len(llm_jobs)]):
            decisions[i] = res
        for (i, _), res in zip(conf_jobs, results[len(llm_jobs):]):
            msg, info = res  # the confederate's words replace the firm's own message; the price is kept
            d = decisions[i]
            decisions[i] = Decision(price=d.price, message=msg, parse_failed=d.parse_failed,
                                    truncated=d.truncated)
            conf_info[str(i)] = info
        prices, sent, flags, fails, truncs = [], [], [], [], []
        for i in range(n):
            if i in forced:
                p = forced[i]
                prices.append(self.bench["nash_prices"][i] if p == "nash" else float(p))
                sent.append(None)
                flags.append(True)
                fails.append(False)
                truncs.append(False)
            else:
                d = decisions[i]
                prices.append(d.price)
                sent.append(d.message)
                flags.append(False)
                fails.append(d.parse_failed)
                truncs.append(d.truncated)
        rnd = self._finalize(t, prices, sent, flags, fails, truncs)
        if conf_info:  # only in rounds with a confederate, so other games' rounds are unchanged
            rnd["confederate"] = conf_info
        self.rounds.append(rnd)
        self._append("rounds.jsonl", rnd)

    def _write_summary(self):
        conf_cost, conf_calls, conf_fallbacks = 0.0, 0, 0
        cpath = self.out / "confederate.jsonl"
        if cpath.exists():
            for c in read_jsonl(cpath):
                conf_calls += 1
                conf_cost += c.get("cost_usd") or 0.0
        plant_cost, plant_calls = 0.0, 0
        ppath = self.out / "plant.jsonl"
        if ppath.exists():
            for c in read_jsonl(ppath):
                plant_calls += 1
                plant_cost += c.get("cost_usd") or 0.0
        conf_fallbacks = sum(info.get("fallback", False) for r in self.rounds
                             for info in (r.get("confederate") or {}).values())
        cost, tokens, calls, unpriced, errors = 0.0, {"prompt": 0, "cached": 0, "completion": 0}, 0, 0, 0
        path = self.out / "calls.jsonl"
        if path.exists():
            for c in read_jsonl(path):
                calls += 1
                errors += "error" in c
                if c.get("cost_usd") is None and route(c.get("model", "")) in ("openai", "anthropic"):
                    unpriced += 1
                cost += c.get("cost_usd") or 0.0
                u = c.get("usage") or {}
                tokens["prompt"] += u.get("prompt_tokens", 0) or 0
                tokens["cached"] += ((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
                tokens["completion"] += u.get("completion_tokens", 0) or 0
        own = own_rounds(self.rounds, self.parent)  # a fork's counts exclude the shared trunk rounds
        llm_idx = [i for i, f in enumerate(self.firms) if isinstance(f, LLMFirm)]
        decisions = sum(1 for r in own for i in llm_idx if not r["forced"][i])
        fails = sum(r["parse_failed"][i] for r in own for i in llm_idx)
        truncs = sum((r.get("truncated") or [False] * len(r["prices"]))[i] for r in own for i in llm_idx)
        summary = {"finished": _now(), "periods": self.rounds[-1]["period"], "llm_calls": calls,
                   "cost_usd": round(cost, 6), "unpriced_openai_calls": unpriced, "call_errors": errors,
                   "tokens": tokens, "llm_decisions": decisions, "parse_failures": fails,
                   "parse_failure_rate": round(fails / decisions, 4) if decisions else None,
                   "truncated_decisions": truncs,
                   "truncation_rate": round(truncs / decisions, 4) if decisions else None,
                   "error": self.error}
        if plant_calls:  # an LLM-driven plant's calls are logged and costed separately (plant.jsonl)
            summary.update({"plant_calls": plant_calls, "plant_cost_usd": round(plant_cost, 6)})
        if conf_calls:  # the confederate's calls are logged and costed separately (confederate.jsonl)
            summary.update({"confederate_calls": conf_calls, "confederate_cost_usd": round(conf_cost, 6),
                            "confederate_fallback_rounds": conf_fallbacks})
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))


_WARNED_UNPRICED: set[str] = set()


def own_rounds(rounds: list[dict], parent: dict | None) -> list[dict]:
    """Rounds this game played itself: for a fork, the periods after the fork point."""
    start = (parent or {}).get("at_period")
    return [r for r in rounds if r["period"] >= 1 and (start is None or r["period"] > start)]


def apply_injections(rnd: dict, interventions: list[dict]):
    """Apply this round's inject_message and notice interventions (each once per round). A round
    without notices has no "notices" key, so prompts of games without notices are unchanged."""
    n = len(rnd["prices"])
    for iv in interventions:
        if iv["type"] not in ("inject_message", "notice") or iv["period"] != rnd["period"] or iv in rnd["injected"]:
            continue
        if iv["type"] == "notice":
            to = list(range(n)) if iv.get("to", "all") == "all" else iv["to"]
            notices = rnd.setdefault("notices", [[] for _ in range(n)])
            for j in to:
                notices[j].append(iv["text"])
        else:
            to = [j for j in range(n) if j != iv["from"]] if iv.get("to", "all") == "all" else iv["to"]
            for j in to:
                rnd["inbox"][j][str(iv["from"])] = iv["text"]
        rnd["injected"].append(iv)


# ---------- loading, resuming, forking ----------
LOGS = ("rounds.jsonl", "calls.jsonl", "confederate.jsonl", "plant.jsonl")  # append-only, resumed in place


def read_jsonl(path: Path, repair: bool = False) -> list[dict]:
    """The records of an append-only log. A last line that does not parse is an interrupted append (the process died
    mid-write): it is skipped with a warning and, with repair, cut from the file so the next append starts on a
    clean line (otherwise it would glue onto the cut record). A bad line anywhere else is corruption and raises."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    out, cut = [], False
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if n < len(lines):
                raise ValueError(f"{path}:{n} is not valid JSON (a corrupt log, not an interrupted append)") from None
            print(f"warning: {path}: ignoring a cut-off last line (interrupted append)"
                  + ("; removed it from the file" if repair else ""), file=sys.stderr, flush=True)
            if repair:
                path.write_text("".join(x + "\n" for x in lines[:-1]))
                cut = True
    if repair and not cut and text and not text.endswith("\n"):  # a whole last record without its newline: end it,
        with path.open("a") as f:                                  # so the next append starts on its own line
            f.write("\n")
    return out


def load_rounds(game_dir: Path, upto: int | None = None) -> list[dict]:
    rounds = read_jsonl(Path(game_dir) / "rounds.jsonl")
    if upto is not None:
        rounds = [r for r in rounds if r["period"] <= upto]
    return rounds


def restore(game_dir: Path, upto: int | None, out_dir: Path, cfg_updates: dict | None = None,
            llm: LLM | None = None, parent: dict | None = None, overwrite: bool = False) -> Game:
    meta = json.loads((Path(game_dir) / "metadata.json").read_text())
    cfg = copy.deepcopy(meta["config"])
    pv = meta.get("prompt_version", 1)
    if pv != PROMPT_VERSION and cfg.get("prompt_mode") == "chat" and len(cfg["firms"]) > 2:
        raise ValueError(f"{game_dir} was played with prompt v{pv}; continuing it with v{PROMPT_VERSION} would change "
                         "the firm names mid-game")
    if cfg_updates:
        cfg.update(cfg_updates)
    if Path(out_dir).resolve() == Path(game_dir).resolve():  # in-place resume: appends follow, so cut a torn last line
        for name in LOGS:
            read_jsonl(Path(game_dir) / name, repair=True)
    all_rounds = load_rounds(game_dir)
    periods = [r["period"] for r in all_rounds]
    if periods != list(range(len(periods))):
        raise ValueError(f"{game_dir}/rounds.jsonl is not one game (periods {periods[:3]}...{periods[-3:]})")
    if upto is not None and not (0 <= upto <= periods[-1]):
        raise ValueError(f"cannot restore {game_dir} at period {upto}: it has periods 0..{periods[-1]}")
    rounds = load_rounds(game_dir, upto)
    g = Game(cfg, out_dir, llm=llm, parent=parent, overwrite=overwrite)
    if parent is None and meta.get("parent"):
        g.parent = meta["parent"]  # in-place resume of a fork keeps its fork point
    g.rounds = rounds
    g.rng = _rng_load(rounds[-1]["rng_state"])
    g.chat_delivery = rounds[-1]["chat_delivery"]
    same_dir = Path(out_dir).resolve() == Path(game_dir).resolve()
    if not same_dir:
        g._write_metadata()
        for r in rounds:
            apply_injections(r, cfg["interventions"])
            g._append("rounds.jsonl", r)
    return g


async def fork(src: Path, at: int, branches: int, interventions: list[dict], out_root: Path,
               label: str, until: int | None = None, llm: LLM | None = None, concurrency: int = 4,
               reseed: bool = False, overwrite: bool = False) -> list[Path]:
    llm = llm or LLM()
    meta = json.loads((Path(src) / "metadata.json").read_text())
    validate_interventions(interventions, len(meta["config"]["firms"]), until or meta["config"]["periods"],
                           fork_at=at)
    base_iv = meta["config"].get("interventions", [])
    dirs, games = [], []
    for b in range(branches):
        out = Path(out_root) / f"{label}-b{b}"
        parent = {"src": str(src), "at_period": at, "branch": b, "interventions": interventions}
        g = restore(src, at, out, {"interventions": base_iv + interventions, "label": label}, llm=llm,
                    parent=parent, overwrite=overwrite)
        if reseed:
            g.rng = random.Random(f"{meta['config']['seed']}-{label}-{b}")
        dirs.append(out)
        games.append(g)
    sem = asyncio.Semaphore(concurrency)

    async def go(g):
        async with sem:
            await g.run(until)

    results = await asyncio.gather(*(go(g) for g in games), return_exceptions=True)
    report_failures(games, results)
    return dirs


def report_failures(games: list, results: list):
    """One game's exception must not kill the others; report each failure (summary.json holds the error)."""
    failed = [(g, r) for g, r in zip(games, results) if isinstance(r, BaseException)]
    for g, r in failed:
        print(f"FAILED {g.out}: {r!r} (resume in place to continue)", file=sys.stderr)
    return failed
