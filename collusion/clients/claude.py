"""Claude models through the official Anthropic SDK (llm.route sends claude-* ids here).

Matched to our other models: adaptive thinking at the given reasoning effort (required: with none the API would
think at its own default, which differs by model), temperature left at the API default 1.0 (the value our other
models run at). Non-default sampling is a 400 on Haiku 5.5 / Sonnet 5.5, so no sampling field is sent; note that
Haiku 5.5's fixed top_p is 0.99, where our open models' no-chat runs use top_p 1.0.
Deliberately off:
- server-side refusal fallbacks: they would answer with a different model and break comparability; a
  refusal comes back as an empty reply, which the firm retries and logs like any unparseable reply;
- prompt caching: our prompt is one growing user message, so cache writes (1.25x input) would never be read.
Usage is mapped to chat-completions keys so costs.py and the game summaries work unchanged.
"""

from __future__ import annotations

import time

from .llm import Reply

EFFORTS = ("low", "medium", "high", "xhigh", "max")
# chat() raises on a stream that ended without a stop reason (a network artifact); reply_of() still treats it as
# cut off, never as a finished answer.
FINISH = {"max_tokens": "length", "model_context_window_exceeded": "length", None: "length", "refusal": "refusal"}


def request(model: str, messages: list[dict], temperature: float | None, max_tokens: int,
            top_p: float | None = None, reasoning_effort: str | None = None,
            reasoning_summary: str | None = None) -> dict:
    """Keyword arguments for messages.stream()."""
    if temperature not in (None, 1.0) or top_p not in (None, 1.0):  # p0 sets top_p 1.0; not sent (see above)
        raise ValueError(f"{model}: Claude models only run at the default sampling (temperature 1.0); "
                         f"got temperature={temperature!r}, top_p={top_p!r}")
    if reasoning_effort not in EFFORTS:
        raise ValueError(f"{model}: set reasoning_effort to one of {EFFORTS} (got {reasoning_effort!r})")
    system = [m["content"] for m in messages if m["role"] == "system"]
    kw = {"model": model, "max_tokens": max_tokens,
          "messages": [m for m in messages if m["role"] != "system"],
          "thinking": {"type": "adaptive", "display": "summarized" if reasoning_summary else "omitted"},
          "output_config": {"effort": reasoning_effort}}
    if system:
        kw["system"] = "\n\n".join(system)
    return kw


def reply_of(msg, latency: float) -> Reply:
    text = "".join(b.text for b in msg.content if b.type == "text")
    thought = "\n\n".join(b.thinking for b in msg.content if b.type == "thinking" and b.thinking) or None
    u = msg.usage
    cached = u.cache_read_input_tokens or 0
    usage = {"prompt_tokens": u.input_tokens + cached + (u.cache_creation_input_tokens or 0),
             "completion_tokens": u.output_tokens,  # includes thinking
             "prompt_tokens_details": {"cached_tokens": cached},
             "completion_tokens_details": {"reasoning_tokens": getattr(u.output_tokens_details, "thinking_tokens", 0)
                                           or 0},
             "stop_reason": msg.stop_reason}
    if msg.stop_reason == "refusal":
        usage["refusal_category"] = getattr(msg.stop_details, "category", None)
        text, thought = "", None  # never parse a price out of a declined reply (lcartel mode also reads reasoning)
    return Reply(content=text, reasoning=thought, usage=usage,
                 finish_reason=FINISH.get(msg.stop_reason, "stop"), latency_s=latency,
                 model_reported=getattr(msg, "model", None))


async def chat(client, model: str, messages: list[dict], temperature: float | None, max_tokens: int,
               top_p: float | None = None, reasoning_effort: str | None = None,
               reasoning_summary: str | None = None) -> Reply:
    kw = request(model, messages, temperature, max_tokens, top_p, reasoning_effort, reasoning_summary)
    t0 = time.monotonic()
    async with client.messages.stream(**kw) as stream:  # streaming: long thinking can outlast a plain request
        try:
            msg = await stream.get_final_message()
        except AssertionError:  # SDK 2.0: the stream ended before any message started
            msg = None
    if msg is None or msg.stop_reason is None:  # the connection closed early: resumable, not a model reply
        raise ConnectionError(f"{model}: the stream ended before the reply finished")
    return reply_of(msg, time.monotonic() - t0)
