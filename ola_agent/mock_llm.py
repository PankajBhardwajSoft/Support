"""Deterministic mock LLM for CrewAI.

Extends CrewAI's documented extension point `crewai.llms.base_llm.BaseLLM`
so no call ever leaves the process: no API key, no network.

How it behaves (ReAct text format, the same one CrewAI asks real models for):
1. If the agent has tools and no tool has run yet, it emits one
   `Action:` / `Action Input:` block for a tool whose arguments it can fill.
2. After a tool has run, it emits `Final Answer:` built from that tool's result.
3. With no usable tool, it emits `Final Answer:` built by `responder`.

Two pitfalls from the brief are handled on purpose:
- CrewAI's own system prompt contains the example line
  "Observation: the result of the action". Tool results are read ONLY from
  assistant-role messages (the model's own turns, to which CrewAI appends the
  real observation), never from system or user messages.
- Tool arguments are chosen from each tool's declared argument schema
  (its field names), never from the tool's name.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr

OBSERVATION_MARKER = "Observation:"

# Ticket IDs look like TKT-0001 (PRD decision D6).
RECORD_ID_PATTERN = re.compile(r"\bTKT-\d{4}\b")

# The user message CrewAI builds starts the task text after this label.
_TASK_LABEL = "Current Task:"
_TASK_END_LABELS = ("This is the expected criteria", "\n\nBegin!")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _normalize(messages: str | list[dict[str, Any]]) -> list[dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [{"role": m.get("role", ""), "content": _content_text(m.get("content"))} for m in messages]


def extract_task_text(messages: list[dict[str, str]]) -> str:
    """Return the task the agent was given, from the first user message."""
    for m in messages:
        if m["role"] != "user":
            continue
        text = m["content"]
        start = text.find(_TASK_LABEL)
        if start == -1:
            return text.strip()
        text = text[start + len(_TASK_LABEL):]
        for label in _TASK_END_LABELS:
            end = text.find(label)
            if end != -1:
                text = text[:end]
        return text.strip()
    return ""


def extract_observations(messages: list[dict[str, str]]) -> list[str]:
    """Return real tool results, read only from the model's own (assistant) turns.

    System and user messages are skipped on purpose: CrewAI's template text
    "Observation: the result of the action" lives there and is not a result.
    """
    results = []
    for m in messages:
        if m["role"] != "assistant":
            continue
        text = m["content"]
        idx = text.rfind(OBSERVATION_MARKER)
        if idx != -1:
            results.append(text[idx + len(OBSERVATION_MARKER):].strip())
    return results


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema_cls = getattr(tool, "args_schema", None)
    if schema_cls is None:
        return {}
    return schema_cls.model_json_schema()


def build_tool_args(schema: dict[str, Any], task_text: str) -> dict[str, Any] | None:
    """Fill a tool's arguments from its declared schema, or None if we can't.

    Dispatch is by the schema's field names only:
    - `record_id`: the first ticket ID (TKT-0000 format) found in the task.
    - `query` / `question`: the task text itself.
    A required field we don't know how to fill means this tool is not used.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", list(properties))
    args: dict[str, Any] = {}
    for field in properties:
        if field == "record_id":
            match = RECORD_ID_PATTERN.search(task_text)
            if match:
                args[field] = match.group(0)
        elif field in {"query", "question"}:
            if task_text:
                args[field] = task_text
    if any(field not in args for field in required):
        return None
    return args


def _default_responder(task_text: str, observations: list[str]) -> str:
    if observations:
        return observations[-1]
    return task_text


class MockLLM(BaseLLM):
    """Deterministic, offline LLM for CrewAI. Same input always gives same output."""

    model: str = "mock-llm"
    # Hook for later steps (e.g. the Composer) to shape the final answer.
    responder: Callable[[str, list[str]], str] = _default_responder
    _call_count: int = PrivateAttr(default=0)

    def __init__(self, **data: Any) -> None:
        # BaseLLM's validator requires `model` in the input, not just a default.
        data.setdefault("model", "mock-llm")
        super().__init__(**data)

    @property
    def call_count(self) -> int:
        return self._call_count

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> str:
        self._call_count += 1
        msgs = _normalize(messages)
        task_text = extract_task_text(msgs)
        observations = extract_observations(msgs)

        if not observations:
            agent_tools = list(getattr(from_agent, "tools", None) or [])
            for tool in agent_tools:
                args = build_tool_args(_tool_schema(tool), task_text)
                if args is not None:
                    return (
                        f"Thought: I need to use a tool to answer this.\n"
                        f"Action: {tool.name}\n"
                        f"Action Input: {json.dumps(args)}"
                    )

        answer = self.responder(task_text, observations)
        return f"Thought: I now know the final answer\nFinal Answer: {answer}"

    def supports_function_calling(self) -> bool:
        # Forces CrewAI's text ReAct loop, which this mock implements.
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192
