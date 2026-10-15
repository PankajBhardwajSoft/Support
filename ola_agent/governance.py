"""AI governance (Part 4, Task 15).

Application layer, least autonomy: `ALLOWED_TOOLS` says which agent may hold
which tool. `check_tool_permissions` runs every time the crew is built, so
handing the ticket tool to any agent other than the Lookup Agent stops the
crew before it can run.

Runtime layer, budget cap (PRD D16): every request's size is estimated
without a tokenizer, and a request over `MAX_REQUEST_TOKENS` is rejected
before it reaches the cache, the crew or a real model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

ALLOWED_TOOLS: dict[str, set[str]] = {
    "Retrieval Agent": {"search_ola_policies"},
    "Lookup Agent": {"check_support_ticket_status"},
    "Response Composer": set(),
}

MAX_REQUEST_TOKENS = 2000
BUDGET_REFUSAL = (f"Sorry, that request is too long. Please keep your question under about "
                  f"{MAX_REQUEST_TOKENS} tokens (roughly {MAX_REQUEST_TOKENS * 3 // 4} words).")

_WORDS = re.compile(r"\S+")


def check_tool_permissions(agents) -> None:
    """Raise PermissionError if an agent holds a tool it is not allowed, or is not listed at all."""
    for agent in agents:
        allowed = ALLOWED_TOOLS.get(agent.role)
        if allowed is None:
            raise PermissionError(f"agent '{agent.role}' is not in the governance table")
        extra = sorted({t.name for t in (agent.tools or [])} - allowed)
        if extra:
            raise PermissionError(f"agent '{agent.role}' may not use {', '.join(extra)}")


def estimate_tokens(text: str) -> int:
    """About 1.3 tokens per word or 1 token per 4 characters, whichever is larger.

    Characters are counted as sent (invisible ones included), so padding a
    request cannot make it look smaller.
    """
    text = text or ""
    return max(math.ceil(len(_WORDS.findall(text)) * 1.3), math.ceil(len(text) / 4))


@dataclass
class BudgetCheck:
    tokens: int
    ok: bool
    reason: str | None = None


def check_budget(text: str, limit: int = MAX_REQUEST_TOKENS) -> BudgetCheck:
    tokens = estimate_tokens(text)
    if tokens > limit:
        return BudgetCheck(tokens, False, f"about {tokens} tokens, over the {limit}-token cap")
    return BudgetCheck(tokens, True)
