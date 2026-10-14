"""The CrewAI crew (Part 2, Task 7): Retrieval Agent, Lookup Agent, Response Composer.

Every agent runs on MockLLM (no network, no keys). Tasks run in order; the
Composer receives both earlier results as task context and writes one draft
that must match `SupportResponse` (Task 9). With MOCK_LLM=false, only the
Composer uses the real Groq model; the other two agents stay on MockLLM so
the tool calls stay predictable.

Least autonomy: only the Lookup Agent is given `check_support_ticket_status`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from crewai import Agent, Crew, Process, Task

from ola_agent import config
from ola_agent.guardrails import join_passages, ticket_sentence
from ola_agent.mock_llm import MockLLM, _normalize
from ola_agent.schema import SupportResponse, TicketInfo
from ola_agent.tools.policy_search import PolicySearchTool
from ola_agent.tools.ticket_lookup import TicketLookupTool

NO_TICKET = {"found": False, "record_id": None, "message": "No ticket ID in the question."}

_CONTEXT_LABEL = "This is the context you're working with:"
_CONTEXT_END = "Provide your complete response"
_CONTEXT_SEPARATOR = "----------"

# Added to the Composer's task only for the real model.
REAL_COMPOSER_RULES = (
    " Use only the policy passages and the ticket result given to you as context. "
    "Reuse their wording; do not add facts, numbers or phone numbers of your own. "
    "If they do not answer the question, reply exactly: I don't know."
)


def _lookup_responder(task_text: str, observations: list[str]) -> str:
    # The Lookup Agent reports the tool result, or that there was no ticket to look up.
    return observations[-1] if observations else json.dumps(NO_TICKET)


def _context_blocks(user_text: str) -> list[dict[str, Any]]:
    """JSON results of earlier tasks, from the Composer's own task prompt."""
    start = user_text.find(_CONTEXT_LABEL)
    if start == -1:
        return []
    text = user_text[start + len(_CONTEXT_LABEL):]
    end = text.find(_CONTEXT_END)
    if end != -1:
        text = text[:end]
    blocks = []
    for piece in text.split(_CONTEXT_SEPARATOR):
        piece = piece.strip()
        try:
            value = json.loads(piece)
        except ValueError:
            continue
        if isinstance(value, dict):
            blocks.append(value)
    return blocks


def compose_draft(blocks: list[dict[str, Any]]) -> SupportResponse:
    """Combine the policy passages and the ticket result into one draft reply.

    The draft uses every retrieved passage; the output guardrail later drops
    what the evidence does not support.
    """
    search = next((b for b in blocks if "chunks" in b), {"chunks": []})
    lookup = next((b for b in blocks if "found" in b), NO_TICKET)
    chunks = search["chunks"]

    ticket = TicketInfo(**lookup) if lookup.get("record_id") else None
    parts = []
    if ticket is not None:
        parts.append(ticket_sentence(ticket))
    if chunks:
        parts.append(join_passages(chunks))
    top = max((c["similarity"] for c in chunks), default=0.0)
    return SupportResponse(
        answer=" ".join(parts) or "No information was found.",
        sources=list(dict.fromkeys(c["doc_id"] for c in chunks)),
        ticket=ticket,
        grounded=bool(parts),
        confidence=min(max(top, 0.0), 1.0),
    )


class ComposerLLM(MockLLM):
    """MockLLM for the Composer: reads the earlier task results from its own prompt."""

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> str:
        self._call_count += 1
        msgs = _normalize(messages)
        user_text = next((m["content"] for m in msgs if m["role"] == "user"), "")
        draft = compose_draft(_context_blocks(user_text))
        return f"Thought: I now know the final answer\nFinal Answer: {draft.model_dump_json()}"


@dataclass
class CrewRun:
    draft: SupportResponse
    search: dict[str, Any]
    lookup: dict[str, Any]
    policy_tool_calls: list[str]
    ticket_tool_calls: list[str]
    agents: dict[str, Agent]


def build_crew(question: str, client=None) -> tuple[Crew, PolicySearchTool, TicketLookupTool]:
    policy_tool = PolicySearchTool(client=client)
    ticket_tool = TicketLookupTool()

    retrieval = Agent(
        role="Retrieval Agent",
        goal="Find the Ola support policy passages that answer the customer's question.",
        backstory="You search Ola's support policy documents.",
        tools=[policy_tool],
        llm=MockLLM(),
        allow_delegation=False,
    )
    lookup = Agent(
        role="Lookup Agent",
        goal="Look up the ticket the customer mentions, if any.",
        backstory="You are the only agent allowed to read ticket records.",
        tools=[ticket_tool],
        llm=MockLLM(responder=_lookup_responder),
        allow_delegation=False,
    )
    composer = Agent(
        role="Response Composer",
        goal="Combine the policy passages and the ticket result into one reply.",
        backstory="You write the reply the customer receives.",
        tools=[],
        llm=ComposerLLM() if config.MOCK_LLM else _real_llm(),
        allow_delegation=False,
    )
    # The question is the whole task description, so the tools receive it unchanged.
    search_task = Task(description=question, expected_output="Policy passages as JSON.", agent=retrieval)
    lookup_task = Task(description=question, expected_output="Ticket lookup result as JSON.", agent=lookup)
    compose_task = Task(
        description=f"Write one reply to the customer question: {question}"
        + ("" if config.MOCK_LLM else REAL_COMPOSER_RULES),
        expected_output="A reply matching the SupportResponse schema.",
        agent=composer,
        context=[search_task, lookup_task],
        output_pydantic=SupportResponse,
    )
    crew = Crew(agents=[retrieval, lookup, composer], tasks=[search_task, lookup_task, compose_task],
                process=Process.sequential, verbose=False)
    return crew, policy_tool, ticket_tool


def run_crew(question: str, client=None) -> CrewRun:
    crew, policy_tool, ticket_tool = build_crew(question, client)
    output = crew.kickoff()
    # Task 9: validate in code, not only through CrewAI's conversion.
    draft = SupportResponse.model_validate(output.pydantic.model_dump() if output.pydantic else json.loads(output.raw))
    search_raw, lookup_raw = output.tasks_output[0].raw, output.tasks_output[1].raw
    search, lookup = _as_json(search_raw, {"chunks": []}), _as_json(lookup_raw, NO_TICKET)
    if not config.MOCK_LLM:
        # The ticket and confidence come from the tools, never from the model's text.
        top = max((c["similarity"] for c in search.get("chunks", [])), default=0.0)
        draft = draft.model_copy(update={
            "ticket": TicketInfo(**lookup) if lookup.get("record_id") else None,
            "confidence": min(max(top, 0.0), 1.0),
        })
    return CrewRun(
        draft=draft,
        search=search,
        lookup=lookup,
        policy_tool_calls=list(policy_tool.calls),
        ticket_tool_calls=list(ticket_tool.calls),
        agents={a.role: a for a in crew.agents},
    )


def _real_llm():
    from ola_agent.real_llm import groq_llm

    return groq_llm()


def _as_json(raw: str, default: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except ValueError:
        return default
    return value if isinstance(value, dict) else default
