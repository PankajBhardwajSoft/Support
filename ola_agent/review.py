"""Autogen review stage (Part 4, Task 14).

A two-agent RoundRobinGroupChat checks the answer after the groundedness check:

- Policy-Compliance-Reviewer: checks every sentence of the draft against the
  retrieved passages and the ticket result (the same sentence rule as the
  Groq-mode groundedness check) and lists the ones they do not support.
- Final-Editor: returns a `Verdict`. Nothing listed: approved, answer unchanged.
  Otherwise: not approved, and the answer keeps only the supported sentences,
  or "I don't know" if none is left. The editor only removes text; it never adds any.

Both agents run on `MockChatClient` in every mode (PRD C4), so the review is
deterministic and needs no key.
"""

from __future__ import annotations

import asyncio
import json
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import LLMMessage
from pydantic import BaseModel, ConfigDict

from ola_agent.guardrails import supported_sentences
from ola_agent.mock_autogen import MockChatClient
from ola_agent.rag.generate import FALLBACK_ANSWER

REVIEWER = "Policy_Compliance_Reviewer"
EDITOR = "Final_Editor"


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    final_answer: str
    reason: str


def review_task(draft: str, evidence: list[str]) -> str:
    """The one task message: the Composer's draft plus the original evidence."""
    return json.dumps({"draft": draft, "evidence": evidence})


def _task(messages: Sequence[LLMMessage]) -> dict:
    # The team's task is the first message after the system prompt.
    first = next(m for m in messages if getattr(m, "source", None) == "user")
    return json.loads(first.content)


def _reviewer_responder(messages: Sequence[LLMMessage]) -> str:
    task = _task(messages)
    kept, cut = supported_sentences(task["draft"], task["evidence"])
    return json.dumps({"supported": kept, "unsupported": cut})


def _editor_responder(messages: Sequence[LLMMessage]) -> str:
    task = _task(messages)
    report = json.loads(next(m for m in reversed(messages) if getattr(m, "source", None) == REVIEWER).content)
    if not report["unsupported"]:
        verdict = Verdict(approved=True, final_answer=task["draft"],
                          reason=f"all {len(report['supported'])} sentence(s) are supported by the evidence")
    else:
        verdict = Verdict(
            approved=False,
            final_answer=" ".join(report["supported"]) or FALLBACK_ANSWER,
            reason=f"removed {len(report['unsupported'])} sentence(s) the evidence does not support: "
                   + " | ".join(report["unsupported"]),
        )
    return verdict.model_dump_json()


def build_team() -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        REVIEWER,
        model_client=MockChatClient(_reviewer_responder),
        description="Checks each sentence of the draft against the retrieved Ola policy passages.",
        system_message="List every sentence of the draft that the evidence does not support.",
    )
    editor = AssistantAgent(
        EDITOR,
        model_client=MockChatClient(_editor_responder),
        description="Writes the final answer and the verdict.",
        system_message="Approve the draft unchanged, or remove the sentences the reviewer listed. Never add text.",
        output_content_type=Verdict,
    )
    # Without custom_message_types the team cannot carry the editor's StructuredMessage and crashes.
    return RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[Verdict]])


async def areview(draft: str, evidence: list[str]) -> Verdict:
    result = await build_team().run(task=review_task(draft, evidence))
    last = result.messages[-1]
    if not isinstance(last, StructuredMessage) or not isinstance(last.content, Verdict):
        raise ValueError(f"the review ended without a verdict ({result.stop_reason})")
    return Verdict.model_validate(last.content.model_dump())


def review(draft: str, evidence: list[str]) -> Verdict:
    """Run the review from synchronous code (the pipeline runs in a worker thread, never in the event loop)."""
    return asyncio.run(areview(draft, evidence))
