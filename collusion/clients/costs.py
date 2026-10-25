"""Per-call cost from token usage and collusion/clients/prices.json."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

PRICES_FILE = Path(__file__).resolve().parent / "prices.json"
_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")


@lru_cache(maxsize=1)
def prices() -> dict:
    return json.loads(PRICES_FILE.read_text())["usd_per_1m_tokens"]


def price_entry(model: str) -> dict | None:
    """Price row for a model id; dated snapshots (gpt-4.1-2025-04-14) fall back to their base id."""
    p = prices()
    return p.get(model) or p.get(_SNAPSHOT.sub("", model))


def call_cost(model: str, usage: dict | None) -> float | None:
    """USD cost of one call, or None if the model has no price entry (e.g. self-hosted)."""
    if not usage:
        return 0.0
    p = price_entry(model)
    if p is None:
        return None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    fresh = (usage.get("prompt_tokens", 0) or 0) - cached
    out = usage.get("completion_tokens", 0) or 0
    return (fresh * p["input"] + cached * p["cached_input"] + out * p["output"]) / 1e6
