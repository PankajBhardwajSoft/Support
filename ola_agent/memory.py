"""Session memory with LangChain (Part 2, Task 8). In-process only, per session_id.

Under MOCK_LLM there is no model to resolve "it", so memory supplies the
ticket ID: a follow-up about a ticket ("How long will it take to resolve?")
with no ID of its own uses the last ticket ID in that session's history.
A new session has no history, so the agent asks for the ID instead.

LangChain marks these classes deprecated (LangChainDeprecationWarning); the
brief says that warning is expected.
"""

from __future__ import annotations

import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from ola_agent.guardrails import mask_phones
from ola_agent.mock_llm import RECORD_ID_PATTERN
from ola_agent.schema import SupportResponse

ASK_FOR_TICKET = "Which ticket do you mean? Please share your ticket ID, for example TKT-0001."

_REFERS_TO_TICKET = re.compile(r"\b(it|its|this ticket|that ticket|my ticket|the ticket)\b", re.I)
_TICKET_INTENT = re.compile(r"\b(status|resolv\w*|resolution|escalat\w*|update|progress|ticket)\b", re.I)

_store: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    return _store.setdefault(session_id, InMemoryChatMessageHistory())


def reset_sessions() -> None:
    _store.clear()


def last_ticket_id(history) -> str | None:
    for message in reversed(history):
        ids = RECORD_ID_PATTERN.findall(str(message.content))
        if ids:
            return ids[-1]
    return None


def is_ticket_followup(question: str) -> bool:
    return (not RECORD_ID_PATTERN.search(question)
            and bool(_REFERS_TO_TICKET.search(question)) and bool(_TICKET_INTENT.search(question)))


class ChatSessions:
    """Multi-turn chat on top of the pipeline. `client` is the ChromaDB client (None: ./chroma_db)."""

    def __init__(self, client=None):
        self.client = client
        self._runnable = RunnableWithMessageHistory(
            RunnableLambda(self._respond),
            get_history,
            input_messages_key="question",
            history_messages_key="history",
            output_messages_key="answer",
        )

    def _respond(self, inputs: dict) -> dict:
        from ola_agent.pipeline import ask

        question, history = inputs["question"], inputs["history"]
        if is_ticket_followup(question):
            ticket_id = last_ticket_id(history)
            if ticket_id is None:
                response = SupportResponse(answer=ASK_FOR_TICKET, sources=[], ticket=None, grounded=False, confidence=0.0)
                return {"answer": response.answer, "response": response, "resolved_question": question}
            question = f"{question} ({ticket_id})"
        response = ask(question, self.client).response
        return {"answer": response.answer, "response": response, "resolved_question": question}

    def chat(self, session_id: str, question: str) -> dict:
        # Mask before history stores the turn, so memory never holds a raw phone number.
        masked, _ = mask_phones(question or "")
        return self._runnable.invoke({"question": masked}, config={"configurable": {"session_id": session_id}})
