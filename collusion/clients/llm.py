"""OpenAI-compatible model client with routing to OpenAI or a vLLM server.

Routing by model id:
- gpt-*, o1/o3/o4-*        -> OpenAI (OPENAI_API_KEY; OPENAI_API_KEY_2 when the first is out of credit)
- qwen*, deepseek*, gpt-oss*, gemma*, glm*, muse*, olmo*, mistral-family, phi*, ids with "/" -> vLLM (VLLM_BASE_URL, VLLM_API_KEY)
- claude*                  -> Anthropic (ANTHROPIC_API_KEY; ANTHROPIC_API_KEY_2 likewise), see claude.py
- fake:<price>             -> offline stub for tests and dry runs
Keys come from the repo-root .env and are never printed.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS = {"openai": ("OPENAI_API_KEY", "OPENAI_API_KEY_2"), "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2")}
_REASONING_OPENAI = re.compile(r"^(o\d|gpt-5|gpt-6)")


def _env() -> dict:
    env = dict(dotenv_values(REPO_ROOT / ".env"))
    import os

    for k in (*KEYS["openai"], *KEYS["anthropic"], "VLLM_BASE_URL", "VLLM_API_KEY", "VLLM_MODEL_REVISION"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def route(model: str) -> str:
    m = model.lower()
    if m.startswith("fake"):
        return "fake"
    if m.startswith("gpt-oss") or "/gpt-oss" in m:
        return "vllm"
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("qwen", "deepseek", "gemma", "glm", "muse", "olmo", "ministral", "magistral", "mistral", "phi", "nemotron")) or "/" in m:
        return "vllm"
    raise ValueError(f"don't know where to send model {model!r}")


def is_openai_reasoning(model: str) -> bool:
    return bool(_REASONING_OPENAI.match(model.lower())) and "chat" not in model.lower()


@dataclass
class Reply:
    content: str
    reasoning: str | None
    usage: dict
    finish_reason: str | None
    latency_s: float
    raw: dict = field(default_factory=dict)
    model_reported: str | None = None  # the model id the provider's response names (alias vs snapshot); None offline


class LLM:
    def __init__(self):
        self._clients = {}
        self._env = _env()
        self._slot = {}  # provider -> index into KEYS: moves to the next key when one is out of credit or expired

    def _client(self, kind: str):
        if kind not in self._clients:
            from openai import AsyncOpenAI

            if kind == "anthropic":
                from anthropic import AsyncAnthropic, Timeout

                name = KEYS[kind][self._slot.get(kind, 0)]
                key = self._env.get(name)
                if not key:
                    raise RuntimeError(f"{name} missing from .env")
                self._clients[kind] = AsyncAnthropic(api_key=key, max_retries=5,
                                                      timeout=Timeout(600.0, connect=10.0))  # fail fast on bad Wi-Fi
            elif kind == "openai":
                name = KEYS[kind][self._slot.get(kind, 0)]
                key = self._env.get(name)
                if not key:
                    raise RuntimeError(f"{name} missing from .env")
                self._clients[kind] = AsyncOpenAI(api_key=key, max_retries=5, timeout=600)
            elif kind == "vllm":
                base = self._env.get("VLLM_BASE_URL")
                if not base:
                    raise RuntimeError("VLLM_BASE_URL missing from .env")
                self._clients[kind] = AsyncOpenAI(base_url=base, api_key=self._env.get("VLLM_API_KEY") or "EMPTY",
                                                  max_retries=5, timeout=1800)  # long thinking replies at ~30 streams per GPU
        return self._clients[kind]

    def serving_info(self) -> dict:
        """Non-secret facts about the serving setup, recorded in metadata."""
        return {"vllm_base_url_set": bool(self._env.get("VLLM_BASE_URL")),
                "vllm_model_revision": self._env.get("VLLM_MODEL_REVISION")}

    async def count_prompt_tokens(self, model: str, messages: list[dict]) -> int | None:
        """Prompt length in tokens after the chat template (vLLM /tokenize); None if unavailable."""
        kind = route(model)
        if kind == "fake":
            return sum(len(m["content"]) for m in messages) // 4
        if kind != "vllm":
            return None
        import json as _json
        import urllib.request

        base = self._env["VLLM_BASE_URL"].rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base
        body = _json.dumps({"model": model, "messages": messages, "add_generation_prompt": True}).encode()
        req = urllib.request.Request(f"{root}/tokenize", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._env.get('VLLM_API_KEY') or 'EMPTY'}")
        req.add_header("User-Agent", "bertrand-harness/0.1")  # RunPod's proxy rejects the urllib default

        def _post():
            with urllib.request.urlopen(req, timeout=120) as r:
                return _json.loads(r.read())

        return int((await asyncio.to_thread(_post))["count"])

    async def chat(self, model: str, messages: list[dict], temperature: float | None,
                   max_tokens: int, system_as_user: bool = False, top_p: float | None = None,
                   reasoning_effort: str | None = None, reasoning_summary: str | None = None,
                   chat_template_kwargs: dict | None = None, prompt_cache_key: str | None = None) -> Reply:
        kind = route(model)
        if system_as_user and messages and messages[0]["role"] == "system":
            messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
        if kind == "fake":
            return _fake_reply(model, messages, max_tokens)
        if kind in KEYS:
            return await self._with_key_fallback(kind, model, messages, temperature, max_tokens, top_p,
                                                 reasoning_effort, reasoning_summary, prompt_cache_key)
        return await self._chat_once(kind, model, messages, temperature, max_tokens, top_p, reasoning_effort,
                                     reasoning_summary, chat_template_kwargs, prompt_cache_key)

    async def _with_key_fallback(self, kind, model, messages, temperature, max_tokens, top_p, reasoning_effort,
                                 reasoning_summary, prompt_cache_key) -> Reply:
        """Same call on the provider's next key when the current one is out of credit or expired (never on any
        other error). The switch is for the whole run; usage records which key paid ("key_slot": 1 or 2)."""
        while True:
            slot = self._slot.get(kind, 0)
            try:
                reply = await self._chat_once(kind, model, messages, temperature, max_tokens, top_p,
                                              reasoning_effort, reasoning_summary, None, prompt_cache_key)
            except Exception as e:
                if not out_of_credit(e):
                    raise
                if self._slot.get(kind, 0) == slot:  # the first call to notice switches; the others just retry
                    nxt = slot + 1
                    if nxt >= len(KEYS[kind]) or not self._env.get(KEYS[kind][nxt]):
                        raise
                    print(f"[llm] {KEYS[kind][slot]} is out of credit or expired ({type(e).__name__}); "
                          f"switching to {KEYS[kind][nxt]}", flush=True)
                    self._slot[kind] = nxt
                    self._clients.pop(kind, None)
                continue
            reply.usage["key_slot"] = slot + 1
            return reply

    async def _chat_once(self, kind, model, messages, temperature, max_tokens, top_p, reasoning_effort,
                         reasoning_summary, chat_template_kwargs, prompt_cache_key) -> Reply:
        if kind == "anthropic":
            from . import claude

            return await claude.chat(self._client(kind), model, messages, temperature, max_tokens, top_p,
                                     reasoning_effort, reasoning_summary)
        if kind == "openai" and reasoning_effort:
            return await self._responses(model, messages, temperature, max_tokens, reasoning_effort,
                                         reasoning_summary, prompt_cache_key)

        kwargs = {"model": model, "messages": messages}
        if kind == "openai":
            kwargs["max_completion_tokens"] = max_tokens
            if prompt_cache_key:
                kwargs["prompt_cache_key"] = prompt_cache_key
            if not is_openai_reasoning(model):
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if top_p is not None:
                    kwargs["top_p"] = top_p
        else:
            kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                # Send every sampling field explicitly so the server's generation_config.json
                # (e.g. R1-Distill: top_p 0.95) can't fill them in; matches vLLM SamplingParams defaults.
                kwargs["top_p"] = top_p
                kwargs["extra_body"] = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}
            if chat_template_kwargs:
                # e.g. {"enable_thinking": true} for Gemma 4, whose template defaults to no thinking
                kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = chat_template_kwargs

        t0 = time.monotonic()
        resp = await self._client(kind).chat.completions.create(**kwargs)
        latency = time.monotonic() - t0
        choice = resp.choices[0]
        msg = choice.message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
        usage = resp.usage.model_dump() if resp.usage else {}
        return Reply(content=msg.content or "", reasoning=reasoning, usage=usage,
                     finish_reason=choice.finish_reason, latency_s=latency,
                     model_reported=getattr(resp, "model", None))

    async def _responses(self, model: str, messages: list[dict], temperature: float | None, max_tokens: int,
                         effort: str, summary: str | None, cache_key: str | None = None) -> Reply:
        """OpenAI Responses API, as LLMCartel's chatgpt_duopoly_tester.py: reasoning effort + summary,
        temperature sent as given, and one retry at effort "low" if the reply hits max_output_tokens.
        Usage is mapped to chat-completions keys so costs.py works unchanged."""
        client = self._client("openai")

        async def create(eff):
            kw = {"model": model, "input": messages, "max_output_tokens": max_tokens,
                  "reasoning": {"effort": eff, **({"summary": summary} if summary else {})}}
            if temperature is not None:
                kw["temperature"] = temperature
            if cache_key:  # routes a game's calls to the same cache (the history only grows, so the prefix repeats)
                kw["prompt_cache_key"] = cache_key
            return await client.responses.create(**kw)

        t0 = time.monotonic()
        resp = await create(effort)
        reduced = False
        if getattr(resp, "status", None) == "incomplete" and getattr(
                getattr(resp, "incomplete_details", None), "reason", None) == "max_output_tokens":
            first_usage = resp.usage
            resp = await create("low")
            reduced = True
        latency = time.monotonic() - t0
        text = (getattr(resp, "output_text", "") or "").strip()
        thought = None
        for item in resp.output or []:
            if item.type == "reasoning" and getattr(item, "summary", None):
                thought = item.summary[0].text
                break
        incomplete = getattr(resp, "status", None) == "incomplete" and getattr(
            getattr(resp, "incomplete_details", None), "reason", None) == "max_output_tokens"

        def usage_of(u):
            if not u:
                return {}
            cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
            reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
            return {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached},
                    "completion_tokens_details": {"reasoning_tokens": reasoning}}

        usage = usage_of(resp.usage)
        if reduced:  # both requests are billed; keep the total
            u1 = usage_of(first_usage)
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] = usage.get(k, 0) + u1.get(k, 0)
            usage["prompt_tokens_details"]["cached_tokens"] += u1["prompt_tokens_details"]["cached_tokens"]
            usage["completion_tokens_details"]["reasoning_tokens"] += u1["completion_tokens_details"]["reasoning_tokens"]
        usage["reduced_effort_retry"] = reduced
        return Reply(content=text, reasoning=thought, usage=usage,
                     finish_reason="length" if incomplete else "stop", latency_s=latency,
                     model_reported=getattr(resp, "model", None))


def out_of_credit(e: BaseException) -> bool:
    """The key itself is unusable: no credit left (OpenAI 429 insufficient_quota, Anthropic 402 billing_error or a
    400 "credit balance is too low"), or it was revoked/expired (401). Rate limits and outages are not."""
    status = getattr(e, "status_code", None)
    text = str(e).lower()
    if status in (401, 402) or getattr(e, "type", None) == "billing_error":
        return True
    if status == 429 and (getattr(e, "code", None) == "insufficient_quota" or "insufficient_quota" in text):
        return True
    return status == 400 and "credit balance" in text


def _fake_reply(model: str, messages: list[dict], max_tokens: int | None = None) -> Reply:
    # fake:<price>[:<message>] -> always answers with that price (and message)
    # fake-raw:<text>          -> answers with <text> verbatim ("\\n" becomes a newline); finish_reason "stop"
    # fake-cut:<text>          -> same, but finish_reason "length" (reply cut off by max_tokens)
    head, _, rest = model.partition(":")
    n_in = sum(len(m["content"]) for m in messages) // 4
    usage = {"prompt_tokens": n_in, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0},
             "requested_max_tokens": max_tokens}
    if head in ("fake-raw", "fake-cut"):
        return Reply(content=rest.replace("\\n", "\n"), reasoning=None, usage=usage,
                     finish_reason="length" if head == "fake-cut" else "stop", latency_s=0.0)
    parts = model.split(":", 2)
    price = parts[1] if len(parts) > 1 else "1.5"
    content = f"My chosen price: {price}"
    if len(parts) > 2:
        content += f"\nMy message: {parts[2]}"
    return Reply(content=content, reasoning=None, usage=usage, finish_reason="stop", latency_s=0.0)


async def gather_limited(coros, limit: int):
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))
