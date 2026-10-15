"""FastAPI app (Part 3, Task 11): two HTTP endpoints and one WebSocket chat.

Run it with:  uvicorn ola_agent.api:app
Every HTTP request and every WebSocket message writes one JSON-Lines log
entry with a trace ID (Task 12, ola_agent/request_log.py).
"""

from __future__ import annotations

from ola_agent import config  # noqa: F401  (telemetry off before crewai is imported)

import threading
import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from ola_agent.cache import CACHE
from ola_agent.guardrails import find_injection, mask_phones
from ola_agent.memory import ChatSessions
from ola_agent.pipeline import ask
from ola_agent.rag.chunking import fixed_size_chunks
from ola_agent.rag.generate import RECOMMENDED_COLLECTION
from ola_agent.rag.index import get_collection, upsert_chunks
from ola_agent.rag.kb import Document
from ola_agent.request_log import new_trace_id, write_entry
from ola_agent.schema import TicketInfo

app = FastAPI(title="Ola Support Agent", version="0.4.0")
# ChromaDB client used by every endpoint. None means the default ./chroma_db;
# tests and demos set a temporary one.
app.state.chroma_client = None

WS_ENDPOINT = "WS /ws/chat"
_add_lock = threading.Lock()


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    sources: list[str]
    ticket: TicketInfo | None
    grounded: bool
    confidence: float
    guardrails_fired: list[str]


class AddDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Only new KB-style IDs: no paths, no free text.
    doc_id: str = Field(pattern=r"^KB-\d{2,3}$")
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    collection: str
    chunks_added: int


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


@app.middleware("http")
async def log_each_request(request: Request, call_next):
    """One log line per HTTP request, including ones rejected before the handler runs."""
    trace_id = new_trace_id()
    request.state.trace_id = trace_id
    request.state.log = {}
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Trace-ID"] = trace_id
        return response
    finally:
        write_entry(trace_id=trace_id, endpoint=f"{request.method} {request.url.path}",
                    status=status, duration_ms=_ms(start), **request.state.log)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    result = ask(body.question, request.app.state.chroma_client)
    r = result.response
    request.state.log.update(question=result.question, answer=r.answer, sources=r.sources,
                             grounded=r.grounded, guardrails_fired=result.guardrails_fired)
    if result.over_budget:  # Task 15 runtime layer: rejected before the crew ran
        raise HTTPException(status_code=413, detail=r.answer)
    return AskResponse(trace_id=request.state.trace_id, answer=r.answer, sources=r.sources, ticket=r.ticket,
                       grounded=r.grounded, confidence=r.confidence, guardrails_fired=result.guardrails_fired)


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(body: AddDocumentRequest, request: Request) -> AddDocumentResponse:
    title, _ = mask_phones(body.title)
    text, masked = mask_phones(body.text)
    request.state.log.update(question=f"add {body.doc_id}: {title}",
                             guardrails_fired=[f"input: masked {masked} phone number(s)"] if masked else [])
    hit = find_injection(f"{title} {text}")
    if hit is not None:
        request.state.log["guardrails_fired"].append(f"input: blocked (prompt injection: '{hit}')")
        raise HTTPException(status_code=400, detail="Document rejected: it contains a prompt-injection pattern.")
    doc = Document(doc_id=body.doc_id, title=title, text=" ".join(text.split()))
    collection = get_collection(RECOMMENDED_COLLECTION, request.app.state.chroma_client)
    with _add_lock:
        if collection.get(where={"doc_id": body.doc_id}, limit=1)["ids"]:
            raise HTTPException(status_code=409, detail=f"{body.doc_id} already exists and cannot be overwritten.")
        chunks = fixed_size_chunks(doc)
        upsert_chunks(collection, chunks)
        CACHE.clear()  # cached answers may no longer match the KB
    request.state.log.update(answer=f"added {len(chunks)} chunk(s)", sources=[body.doc_id])
    return AddDocumentResponse(trace_id=request.state.trace_id, doc_id=body.doc_id,
                               collection=RECOMMENDED_COLLECTION, chunks_added=len(chunks))


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Multi-turn chat. Each connection is its own memory session."""
    await websocket.accept()
    session_id = new_trace_id()
    sessions = ChatSessions(websocket.app.state.chroma_client)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            trace_id, start = new_trace_id(), time.perf_counter()
            question = message.get("text")
            if question is None:
                write_entry(trace_id=trace_id, endpoint=WS_ENDPOINT, session_id=session_id,
                            status="rejected: not text", duration_ms=_ms(start))
                await websocket.send_json({"trace_id": trace_id, "session_id": session_id,
                                           "error": "Please send text messages only."})
                continue
            try:
                result = await run_in_threadpool(sessions.chat, session_id, question)
            except Exception as exc:  # one bad message must not end the chat or the app
                write_entry(trace_id=trace_id, endpoint=WS_ENDPOINT, session_id=session_id,
                            status=f"error: {type(exc).__name__}", duration_ms=_ms(start))
                await websocket.send_json({"trace_id": trace_id, "session_id": session_id,
                                           "error": "Sorry, something went wrong. Please try again."})
                continue
            r = result["response"]
            reply = {"trace_id": trace_id, "session_id": session_id, "answer": r.answer,
                     "sources": r.sources, "grounded": r.grounded}
            write_entry(trace_id=trace_id, endpoint=WS_ENDPOINT, session_id=session_id,
                        question=result["resolved_question"], answer=r.answer, sources=r.sources,
                        grounded=r.grounded, status="ok", duration_ms=_ms(start))
            await websocket.send_json(reply)
    except WebSocketDisconnect as exc:
        # The client went away (even mid-answer). Only this connection ends; the app keeps serving others.
        write_entry(trace_id=new_trace_id(), endpoint=WS_ENDPOINT, session_id=session_id,
                    status=f"disconnected (code {exc.code})")
