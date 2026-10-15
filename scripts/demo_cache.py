"""Task 16: a repeated question is served from the cache, with no new model or tool call.

Counters wrap the mock model and both tools for this demo only, so the
before/after numbers are counted, not estimated.
"""

import time

import _bootstrap  # noqa: F401

from ola_agent.cache import CACHE
from ola_agent.crew import ComposerLLM
from ola_agent.mock_llm import MockLLM
from ola_agent.pipeline import ask
from ola_agent.tools.policy_search import PolicySearchTool
from ola_agent.tools.ticket_lookup import TicketLookupTool

COUNTS = {"model calls": 0, "policy tool calls": 0, "ticket tool calls": 0}


def counted(cls, method, label):
    original = getattr(cls, method)

    def wrapper(self, *args, **kwargs):
        COUNTS[label] += 1
        return original(self, *args, **kwargs)

    setattr(cls, method, wrapper)


counted(MockLLM, "call", "model calls")
counted(ComposerLLM, "call", "model calls")
counted(PolicySearchTool, "_run", "policy tool calls")
counted(TicketLookupTool, "_run", "ticket tool calls")


def timed_ask(label, question):
    before = dict(COUNTS)
    start = time.perf_counter()
    r = ask(question)
    ms = (time.perf_counter() - start) * 1000
    new = {k: COUNTS[k] - before[k] for k in COUNTS}
    print(f"{label}: {question!r}")
    print(f"   cache hit: {r.cache_hit}   new calls: {new}   time: {ms:.0f} ms")
    print(f"   answer starts: {r.response.answer[:70]}...")
    return r, new


def main():
    CACHE.clear()
    q = "What is the status of TKT-0023?"
    r1, first = timed_ask("1) first ask", q)
    r2, second = timed_ask("2) same question again", q)
    r3, third = timed_ask("3) same question, other case and spacing", "  what is the STATUS of TKT-0023 ")
    print(f"\ncache entries: {len(CACHE)}, hits: {CACHE.hits}, misses: {CACHE.misses}")
    same = r1.response == r2.response == r3.response
    print(f"same answer all three times: {same}")
    ok = (not r1.cache_hit and first["model calls"] > 0 and first["ticket tool calls"] == 1
          and r2.cache_hit and r3.cache_hit and not any(second.values()) and not any(third.values()) and same)
    print(f"\nCACHE DEMO: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
