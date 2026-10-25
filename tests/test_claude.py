"""Claude client (claude.py): request shape, reply/usage mapping, refusals, costs and a short self-play game,
all against a fake Anthropic client (no API calls)."""

import asyncio
import json

import pytest
from anthropic.types import Message

from collusion.clients import claude
from collusion.experiments import common
from collusion.experiments import selfplay as sp
from collusion.clients.costs import call_cost
from collusion.clients.llm import LLM, route

MSGS = [{"role": "system", "content": "You are firm 1."}, {"role": "user", "content": "Round 3. Your price?"}]


def message(text="My chosen price: 1.8\nMy message: Let's hold.", thought="Rivals at 1.7.", stop="end_turn",
            usage=None, category=None):
    content = ([{"type": "thinking", "thinking": thought, "signature": "sig"}] if thought is not None else []) + (
        [{"type": "text", "text": text}] if text else [])
    u = usage or {"input_tokens": 1000, "output_tokens": 300, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0, "output_tokens_details": {"thinking_tokens": 250}}
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "content": content,
        "stop_reason": stop, "stop_sequence": None, "usage": u,
        "stop_details": {"type": "refusal", "category": category, "explanation": None} if stop == "refusal" else None})


class FakeClient:
    """messages.stream(**kw) as an async context manager whose get_final_message() returns the next reply."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = self

    def stream(self, **kw):
        self.calls.append(kw)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

        class S:
            async def __aenter__(s):
                return s

            async def __aexit__(s, *a):
                return False

            async def get_final_message(s):
                return reply

        return S()


def llm_with(client):
    llm = LLM()
    llm._clients["anthropic"] = client
    return llm


def test_claude_ids_route_to_anthropic():
    assert route("claude-haiku-5-5") == route("claude-sonnet-5-5") == "anthropic"
    assert route("gpt-6-luna") == "openai" and route("gpt-oss-120b") == "vllm"


def test_request_uses_default_sampling_adaptive_thinking_and_effort():
    fake = FakeClient([message()])
    r = asyncio.run(llm_with(fake).chat("claude-haiku-5-5", MSGS, 1.0, 16384, reasoning_effort="medium",
                                        reasoning_summary="auto", prompt_cache_key="g|f0"))
    kw = fake.calls[0]
    assert kw["system"] == "You are firm 1." and kw["messages"] == MSGS[1:]
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kw["output_config"] == {"effort": "medium"} and kw["max_tokens"] == 16384
    assert not {"temperature", "top_p", "cache_control", "fallbacks"} & set(kw)
    assert r.content.startswith("My chosen price: 1.8") and r.reasoning == "Rivals at 1.7."
    assert r.finish_reason == "stop"
    kw2 = claude.request("claude-haiku-5-5", MSGS, None, 100, reasoning_effort="low")
    assert kw2["thinking"]["display"] == "omitted" and kw2["output_config"] == {"effort": "low"}
    with pytest.raises(ValueError):  # no effort: the API would think at its own, model-specific default
        claude.request("claude-haiku-5-5", MSGS, None, 100)
    with pytest.raises(ValueError):  # OpenAI-only values
        claude.request("claude-haiku-5-5", MSGS, None, 100, reasoning_effort="minimal")


def test_non_default_sampling_is_refused():
    with pytest.raises(ValueError):
        claude.request("claude-haiku-5-5", MSGS, 0.7, 100)
    with pytest.raises(ValueError):
        claude.request("claude-haiku-5-5", MSGS, 1.0, 100, top_p=0.95)  # 1.0 (p0) is accepted


def test_usage_and_cost_map_to_chat_completions_keys():
    u = {"input_tokens": 400, "output_tokens": 300, "cache_read_input_tokens": 600,
         "cache_creation_input_tokens": 0, "output_tokens_details": {"thinking_tokens": 250}}
    r = claude.reply_of(message(usage=u), 1.0)
    assert r.usage["prompt_tokens"] == 1000 and r.usage["prompt_tokens_details"]["cached_tokens"] == 600
    assert r.usage["completion_tokens"] == 300 and r.usage["completion_tokens_details"]["reasoning_tokens"] == 250
    # Haiku 5.5: $0.10 input, $0.01 cached, $0.50 output per 1M tokens
    assert call_cost("claude-haiku-5-5", r.usage) == pytest.approx((400 * 0.10 + 600 * 0.01 + 300 * 0.50) / 1e6)
    assert call_cost("claude-sonnet-5-5", r.usage) is not None


def test_cut_off_and_refused_replies():
    r = claude.reply_of(message(stop="max_tokens"), 1.0)
    assert r.finish_reason == "length"
    r = claude.reply_of(message(text="My chosen price: 2.8", stop="refusal", category="general_harms"), 1.0)
    assert r.finish_reason == "refusal" and r.content == ""
    assert r.usage["stop_reason"] == "refusal" and r.usage["refusal_category"] == "general_harms"
    assert claude.reply_of(message(thought=""), 1.0).reasoning is None


def test_selfplay_game_runs_end_to_end_and_counts_cost(tmp_path):
    fake = FakeClient([message()])
    asyncio.run(sp.selfplay("claude-haiku-5-5", tmp_path, games=1, periods=3, seed0=7, chat=True,
                              sampling={"reasoning_effort": "medium", "max_tokens": 4096, "temperature": 1.0},
                              llm=llm_with(fake)))
    g = tmp_path / "sp-s7"
    rounds = [json.loads(x) for x in open(g / "rounds.jsonl")]
    assert all(p == pytest.approx(1.8) for r in rounds[1:] for p in r["prices"])
    s = json.loads((g / "summary.json").read_text())
    assert s["llm_calls"] == 9 and s["unpriced_openai_calls"] == 0 and s["parse_failures"] == 0
    assert s["cost_usd"] == pytest.approx(9 * (1000 * 0.10 + 300 * 0.50) / 1e6, rel=1e-3)


def test_refusals_become_logged_parse_failures(tmp_path):
    fake = FakeClient([message(text="", stop="refusal")])
    asyncio.run(sp.selfplay("claude-haiku-5-5", tmp_path, games=1, periods=2, seed0=8, chat=True,
                              sampling={"reasoning_effort": "medium", "max_tokens": 4096}, llm=llm_with(fake)))
    g = tmp_path / "sp-s8"
    rounds = [json.loads(x) for x in open(g / "rounds.jsonl")]
    assert all(all(r["parse_failed"]) for r in rounds[1:])
    calls = [json.loads(x) for x in open(g / "calls.jsonl")]
    assert calls and all(c["usage"]["stop_reason"] == "refusal" and c["price"] is None for c in calls)


def test_trap_resumes_on_anthropic_network_errors_but_not_on_bad_requests():
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(code, cls):
        return cls(f"HTTP {code}", response=httpx2.Response(code, request=req), body=None)

    assert common._network_error(anthropic.APIConnectionError(request=req))
    assert common._network_error(anthropic.APITimeoutError(request=req))
    assert common._network_error(status(529, anthropic.APIStatusError))
    assert common._network_error(status(429, anthropic.RateLimitError))
    assert not common._network_error(status(400, anthropic.BadRequestError))
    assert not common._network_error(ValueError("temperature"))


def test_review_fixes_stop_reasons_refusal_reasoning_and_no_chat(tmp_path):
    # a stream that ended without a stop reason, or at the context window, counts as cut off (retried)
    assert claude.reply_of(message(stop=None), 1.0).finish_reason == "length"
    assert claude.reply_of(message(stop="model_context_window_exceeded"), 1.0).finish_reason == "length"
    # a refusal never yields a price, not even from the reasoning (lcartel mode parses reasoning too)
    r = claude.reply_of(message(text="", thought="My chosen price: 2.8", stop="refusal"), 1.0)
    assert r.content == "" and r.reasoning is None
    # p0 (no-chat) sends top_p 1.0, the default: accepted and not sent
    assert "top_p" not in claude.request("claude-haiku-5-5", MSGS, 1.0, 100, top_p=1.0, reasoning_effort="medium")
    fake = FakeClient([message(text="My chosen price: 1.6")])
    asyncio.run(sp.selfplay("claude-haiku-5-5", tmp_path, games=1, periods=2, seed0=9, chat=False,
                              sampling={"reasoning_effort": "medium", "max_tokens": 4096}, llm=llm_with(fake)))
    s = json.loads((tmp_path / "sp-s9" / "summary.json").read_text())
    assert s["llm_calls"] == 6 and s["parse_failures"] == 0


def test_trap_resumes_on_an_error_inside_a_200_stream():
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    e = anthropic.APIStatusError("overloaded", response=httpx2.Response(200, request=req),
                                 body={"type": "error", "error": {"type": "overloaded_error", "message": "x"}})
    assert e.type == "overloaded_error" and common._network_error(e)
    bad = anthropic.APIStatusError("bad", response=httpx2.Response(200, request=req),
                                   body={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}})
    assert not common._network_error(bad)


def test_an_empty_stream_is_a_resumable_connection_error():
    class Empty(FakeClient):
        def stream(self, **kw):
            class S:
                async def __aenter__(s):
                    return s

                async def __aexit__(s, *a):
                    return False

                async def get_final_message(s):
                    raise AssertionError  # what SDK 2.0 does when no message_start arrived

            return S()

    with pytest.raises(ConnectionError) as e:
        asyncio.run(llm_with(Empty([])).chat("claude-haiku-5-5", MSGS, None, 100, reasoning_effort="medium"))
    assert common._network_error(e.value)


def test_real_sdk_over_an_offline_transport():
    """The real AsyncAnthropic client over httpx2.MockTransport (dummy key, no API call): the request body the SDK
    sends, a normal reply, and a stream that closes before the reply finished (resumable, never a price)."""
    import httpx2
    from anthropic import AsyncAnthropic

    start = {"type": "message_start", "message": {
        "id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 1, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0}}}
    blocks = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Rivals at 1.7."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "My chosen price: 1.8"}},
        {"type": "content_block_stop", "index": 1}]
    end = [{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None, "stop_details": None},
            "usage": {"output_tokens": 300, "output_tokens_details": {"thinking_tokens": 250}}},
           {"type": "message_stop"}]
    bodies = []

    def client(events):
        def handler(req):
            bodies.append(json.loads(req.content))
            data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data)

        return AsyncAnthropic(api_key="dummy", max_retries=0,
                              http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))

    def chat(events):
        return asyncio.run(claude.chat(client(events), "claude-haiku-5-5", MSGS, 1.0, 4096, 1.0, "medium", "auto"))

    r = chat([start, *blocks, *end])
    assert r.content == "My chosen price: 1.8" and r.reasoning == "Rivals at 1.7." and r.finish_reason == "stop"
    assert r.usage["completion_tokens"] == 300 and r.usage["completion_tokens_details"]["reasoning_tokens"] == 250
    body = bodies[0]
    assert not {"temperature", "top_p", "top_k", "cache_control", "fallbacks"} & set(body)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "medium"} and body["system"] == "You are firm 1."
    for events in ([start, *blocks], []):  # closed before message_delta / before anything arrived
        with pytest.raises(ConnectionError) as e:
            chat(events)
        assert common._network_error(e.value)
