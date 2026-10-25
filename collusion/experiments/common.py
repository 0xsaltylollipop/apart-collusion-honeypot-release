"""Shared helpers of the experiments: finished/last-round checks, the network-error test used to resume, time."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from collusion.game.engine import read_jsonl


def _network_error(e: BaseException) -> bool:
    """Connection / timeout / server errors from the API client (worth resuming), not design or data errors."""
    mod = type(e).__module__.split(".")[0]
    if mod == "anthropic":  # connection/timeout errors have no status; a 400 is a request bug, not the network
        status = getattr(e, "status_code", None)  # an error inside a 200 stream carries its type instead
        return (status is None or status >= 500 or status in (408, 409, 429)
                or getattr(e, "type", None) in ("overloaded_error", "api_error", "rate_limit_error", "timeout_error"))
    return mod in ("openai", "httpx", "httpx2", "httpcore", "ssl") or isinstance(e, (OSError, asyncio.TimeoutError))


def _finished(d: Path, until: int) -> bool:
    s = d / "summary.json"
    if not s.exists():
        return False
    j = json.loads(s.read_text())
    return j.get("error") is None and j.get("periods") == until


def _last_period(d: Path) -> int:
    rounds = read_jsonl(d / "rounds.jsonl")
    return rounds[-1]["period"] if rounds else -1


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rounds(d: Path) -> list[dict]:
    return read_jsonl(d / "rounds.jsonl")
