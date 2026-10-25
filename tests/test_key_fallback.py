"""Second API key takes over when the first is out of credit or expired, and only then."""

import asyncio

import pytest

from collusion.clients.llm import LLM, Reply, out_of_credit


class Err(Exception):
    def __init__(self, status, code=None, type_=None, msg=""):
        super().__init__(msg)
        self.status_code, self.code, self.type = status, code, type_


def test_what_counts_as_out_of_credit():
    assert out_of_credit(Err(429, code="insufficient_quota"))
    assert out_of_credit(Err(402, type_="billing_error"))
    assert out_of_credit(Err(400, msg="Your credit balance is too low to access the Anthropic API"))
    assert out_of_credit(Err(401, type_="authentication_error"))
    assert not out_of_credit(Err(429, code="rate_limit_exceeded"))
    assert not out_of_credit(Err(529, type_="overloaded_error"))
    assert not out_of_credit(Err(400, msg="bad request"))
    assert not out_of_credit(ConnectionError("dns"))


def fake_llm(fail_on_slot0, env=None):
    llm = LLM()
    llm._env = env or {"OPENAI_API_KEY": "k1", "OPENAI_API_KEY_2": "k2"}
    seen = []

    async def once(kind, *a, **kw):
        slot = llm._slot.get(kind, 0)
        seen.append(slot)
        if slot == 0 and fail_on_slot0:
            raise fail_on_slot0
        return Reply("My chosen price: 1.6", None, {}, "stop", 0.0)

    llm._chat_once = once
    return llm, seen


def test_switches_to_the_second_key_once_and_stays():
    llm, seen = fake_llm(Err(429, code="insufficient_quota"))
    r = asyncio.run(llm.chat("gpt-6-luna", [{"role": "user", "content": "x"}], 1.0, 100))
    assert r.usage["key_slot"] == 2 and seen == [0, 1]
    r = asyncio.run(llm.chat("gpt-6-luna", [{"role": "user", "content": "x"}], 1.0, 100))
    assert r.usage["key_slot"] == 2 and seen == [0, 1, 1]


def test_other_errors_do_not_switch_keys():
    llm, seen = fake_llm(Err(429, code="rate_limit_exceeded"))
    with pytest.raises(Err):
        asyncio.run(llm.chat("gpt-6-luna", [{"role": "user", "content": "x"}], 1.0, 100))
    assert llm._slot.get("openai", 0) == 0


def test_no_second_key_raises_the_original_error():
    llm, _ = fake_llm(Err(402, type_="billing_error"), env={"ANTHROPIC_API_KEY": "k1"})
    with pytest.raises(Err):
        asyncio.run(llm.chat("claude-haiku-5-5", [{"role": "user", "content": "x"}], 1.0, 100))
