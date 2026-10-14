"""Step 4 functional tests: FastAPI endpoints, JSON-Lines logging and the 15-query evaluation (Tasks 11 to 13)."""

import json
import os
import tempfile

import pytest

from ola_agent import request_log
from ola_agent.evaluation import TEST_SET, EvalCase, build_judge_prompt, judge, score_from_prompt
from ola_agent.rag.embeddings import MODEL_DIR

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="embedding model not downloaded")


@pytest.fixture
def log_path(monkeypatch):
    path = os.path.join(tempfile.mkdtemp(), "requests.jsonl")
    monkeypatch.setattr(request_log, "LOG_PATH", path)
    return path


@pytest.fixture
def api(log_path):
    from fastapi.testclient import TestClient

    from ola_agent.api import app
    from ola_agent.rag.index import build_indexes, get_client

    client = get_client(tempfile.mkdtemp())
    build_indexes(client=client)
    app.state.chroma_client = client
    yield TestClient(app)
    app.state.chroma_client = None


# --- Task 12: logging -------------------------------------------------------------------------

def test_write_entry_masks_text_and_keeps_ids(log_path):
    request_log.write_entry(trace_id="ab9876543210cd", session_id="9876543210", question="call +91 98765 43210",
                            answer="or 09876543210", sources=["KB-01"])
    (entry,) = request_log.read_entries()
    assert entry["trace_id"] == "ab9876543210cd" and entry["session_id"] == "9876543210"
    assert entry["question"] == "call [PHONE]" and entry["answer"] == "or [PHONE]"
    assert set(entry) == set(request_log.FIELDS)


def test_one_line_per_entry_even_with_newlines(log_path):
    request_log.write_entry(trace_id="t1", question="line one\nline two three")
    with open(log_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["question"] == "line one\nline two three"


# --- Task 11: endpoints -----------------------------------------------------------------------

@needs_model
def test_ask_returns_model_and_trace_id(api, log_path):
    r = api.post("/ask", json={"question": "When does a ticket get escalated to Tier 2?"})
    body = r.json()
    assert r.status_code == 200 and body["sources"][:1] == ["KB-03"]
    assert set(body) == {"trace_id", "answer", "sources", "ticket", "grounded", "confidence", "guardrails_fired"}
    assert r.headers["x-trace-id"] == body["trace_id"]
    (entry,) = request_log.read_entries()
    assert entry["trace_id"] == body["trace_id"] and entry["endpoint"] == "POST /ask"
    assert entry["status"] == 200 and entry["duration_ms"] > 0


@needs_model
def test_add_document_then_ask_finds_it(api):
    doc = {"doc_id": "KB-13", "title": "Lost item policy",
           "text": "If a rider leaves an item in the cab, support connects them with the driver through in-app chat within 2 hours."}
    r = api.post("/add-document", json=doc)
    assert r.status_code == 200 and r.json()["chunks_added"] == 1 and r.json()["collection"] == "kb_fixed"
    r = api.post("/ask", json={"question": "What happens if I leave an item in the cab?"})
    assert r.json()["sources"][:1] == ["KB-13"]


@needs_model
def test_websocket_memory_and_one_log_line_per_message(api, log_path):
    with api.websocket_connect("/ws/chat") as ws:
        ws.send_text("What is the status of TKT-0007?")
        first = ws.receive_json()
        ws.send_text("How long will it take to resolve it?")
        second = ws.receive_json()
    assert "TKT-0007" in first["answer"] and "TKT-0007" in second["answer"]
    assert first["session_id"] == second["session_id"]
    entries = request_log.read_entries()
    assert [e["trace_id"] for e in entries[:2]] == [first["trace_id"], second["trace_id"]]
    assert entries[-1]["status"].startswith("disconnected")


# --- Task 13: evaluation ----------------------------------------------------------------------

def test_test_set_has_15_queries_covering_every_topic():
    assert len(TEST_SET) == 15
    assert sum(c.kind == "out_of_scope" for c in TEST_SET) == 3
    assert {c.expected_doc for c in TEST_SET if c.kind == "in_scope"} == {f"KB-{n:02d}" for n in range(1, 13)}


def test_judge_rules():
    case = EvalCase("Q?", "in_scope", "KB-06", ("8 AM to 10 PM local time", "Monday to Saturday"))
    chunks = [{"text": "Email support and Tier 2 teams work from 8 AM to 10 PM local time, Monday to Saturday."}]
    good = judge(case, chunks[0]["text"], ["KB-06"], chunks)
    assert (good.accuracy, good.grounding, good.completeness, good.safety) == (1.0, 1.0, 1.0, 1.0)
    half = judge(case, "Email support works from 8 AM to 10 PM local time.", ["KB-02", "KB-06"], chunks)
    assert half.accuracy == 0.5 and half.completeness == 0.5
    refused = judge(case, "I don't know. I couldn't find this.", [], chunks)
    assert refused.accuracy == 0 and refused.completeness == 0 and refused.grounding == 1


def test_judge_out_of_scope_and_safety():
    case = EvalCase("What is Uber's refund policy?", "out_of_scope")
    assert judge(case, "I don't know. I couldn't find this.", [], []).safety == 1.0
    answered = judge(case, "Riders get a full refund.", ["KB-04"], [])
    assert answered.accuracy == 0 and answered.safety == 0


def test_judge_reads_only_its_own_prompt():
    prompt = build_judge_prompt(TEST_SET[0], "x", [], [])
    assert prompt.count("### ") == 8 and score_from_prompt(prompt).accuracy == 0


@needs_model
def test_full_evaluation_runs_and_scores_every_query():
    from ola_agent.evaluation import averages, evaluate
    from ola_agent.rag.index import build_indexes, get_client

    client = get_client(tempfile.mkdtemp())
    build_indexes(client=client)
    rows = evaluate(client=client)
    assert len(rows) == 15
    avg = averages(rows)
    assert set(avg) == {"accuracy", "grounding", "completeness", "safety"}
    assert all(0.0 <= v <= 1.0 for v in avg.values())
    assert all(r.scores.safety == 1.0 for r in rows)
