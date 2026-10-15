"""Step 5 (Part 4): Autogen review (Task 14), governance (Task 15), response cache (Task 16)."""

import os
import tempfile

import pytest
from pydantic import ValidationError

from ola_agent.cache import CACHE, cache_key, normalize_query
from ola_agent.evaluation import TEST_SET
from ola_agent.governance import ALLOWED_TOOLS, BUDGET_REFUSAL, MAX_REQUEST_TOKENS, check_budget, estimate_tokens
from ola_agent.rag.embeddings import MODEL_DIR
from ola_agent.rag.generate import FALLBACK_ANSWER
from ola_agent.review import Verdict, review

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="embedding model not downloaded")

REFUND = ("Refunds to cards and UPI usually arrive within 5 to 7 working days, "
          "while refunds to the Ola wallet arrive within 24 hours.")


@pytest.fixture(scope="module")
def client():
    from ola_agent.rag.index import build_indexes, get_client

    c = get_client(tempfile.mkdtemp())
    build_indexes(client=c)
    return c


# --- Task 14: review ---------------------------------------------------------------------------

def test_verdict_model_fields_are_exact():
    assert set(Verdict.model_fields) == {"approved", "final_answer", "reason"}
    with pytest.raises(ValidationError):
        Verdict(approved=True, final_answer="x", reason="y", score=1)


def test_review_approves_a_supported_draft_unchanged():
    v = review(REFUND, [REFUND])
    assert v.approved and v.final_answer == REFUND


def test_review_removes_an_unsupported_sentence():
    v = review(f"Ola also refunds the full fare for every late ride. {REFUND}", [REFUND])
    assert not v.approved and v.final_answer == REFUND and "late ride" in v.reason


def test_review_with_nothing_supported_says_i_dont_know():
    v = review("Every rider gets a free ride voucher.", [REFUND])
    assert not v.approved and v.final_answer == FALLBACK_ANSWER


@needs_model
def test_pipeline_review_approves_every_mock_answer(client):
    # In mock mode the groundedness check rebuilds answers from the KB, so the review must approve them.
    from ola_agent.pipeline import ask

    for case in TEST_SET:
        r = ask(case.question, client)
        if r.verdict is not None:
            assert r.verdict.approved, case.question
            assert r.verdict.final_answer == r.response.answer


# --- Task 15: governance -----------------------------------------------------------------------

def test_allowed_tools_match_the_crew():
    from ola_agent.crew import build_crew

    crew, _, _ = build_crew("What is the status of TKT-0007?")
    assert {a.role: {t.name for t in a.tools} for a in crew.agents} == ALLOWED_TOOLS


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three four") == 6  # 4 words x 1.3, rounded up
    assert estimate_tokens("a" * 400) == 100  # one long word: 400 characters / 4


def test_budget_cap_boundary():
    assert check_budget("x" * (MAX_REQUEST_TOKENS * 4)).ok
    assert not check_budget("x" * (MAX_REQUEST_TOKENS * 4 + 1)).ok


def test_over_budget_request_never_reaches_the_crew(monkeypatch):
    import ola_agent.pipeline as pipeline

    monkeypatch.setattr(pipeline, "run_crew", lambda *a, **k: pytest.fail("crew ran"))
    r = pipeline.ask("refund " * 2000)
    assert r.over_budget and r.response.answer == BUDGET_REFUSAL and not r.response.grounded
    assert any(f.startswith("runtime: budget cap") for f in r.guardrails_fired)
    assert len(CACHE) == 0


@needs_model
def test_websocket_over_budget_gets_the_refusal(client):
    from fastapi.testclient import TestClient

    from ola_agent.api import app

    app.state.chroma_client = client
    try:
        with TestClient(app).websocket_connect("/ws/chat") as ws:
            ws.send_text("refund " * 2000)
            assert ws.receive_json()["answer"] == BUDGET_REFUSAL
            ws.send_text("What is the status of TKT-0023?")
            assert "TKT-0023" in ws.receive_json()["answer"]
    finally:
        app.state.chroma_client = None


# --- Task 16: cache ----------------------------------------------------------------------------

def test_normalize_query():
    assert normalize_query("  What is the P2 SLA?? ") == normalize_query("what is the p2   sla") == "what is the p2 sla"
    assert normalize_query("ｗｈａｔ​ is") == "what is"


def test_cache_key_keeps_exact_ticket_ids():
    assert cache_key("status of TKT-0023") != cache_key("status of tkt-0023")
    assert cache_key("Status of TKT-0023?") == cache_key("status of TKT-0023")


def test_cache_drops_the_oldest_entry():
    from ola_agent.cache import ResponseCache

    c = ResponseCache(max_entries=2)
    c.put(("a",), 1)
    c.put(("b",), 2)
    c.get(("a",))  # a is now the most recent
    c.put(("c",), 3)
    assert c.keys() == [("a",), ("c",)]


@needs_model
def test_repeated_question_is_served_from_the_cache(client, monkeypatch):
    import ola_agent.pipeline as pipeline

    runs = []
    real = pipeline.run_crew
    monkeypatch.setattr(pipeline, "run_crew", lambda q, c=None: runs.append(q) or real(q, c))
    first = pipeline.ask("What is the status of TKT-0023?", client)
    second = pipeline.ask("what is the status of TKT-0023", client)
    assert len(runs) == 1 and not first.cache_hit and second.cache_hit
    assert first.response == second.response
    assert "cache: hit (crew not run)" in second.guardrails_fired
    assert second.verdict is not None  # the review still runs on a hit
