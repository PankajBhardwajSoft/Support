"""Optional Groq mode (change request approved 2026-10-14): composer and judge use the real model.

No test here reaches Groq. A fake model built on MockLLM stands in for it, so
the switch, the sentence check and the judge parsing run offline with no key.
"""

import json
import os
import tempfile

import pytest

from ola_agent import config, request_log
from ola_agent.evaluation import TEST_SET, EvalCase, judge
from ola_agent.guardrails import FALLBACK_ANSWER, PHONE_PATTERN, check_groundedness, supported_sentences
from ola_agent.mock_llm import MockLLM
from ola_agent.rag.embeddings import MODEL_DIR
from ola_agent.schema import SupportResponse

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="embedding model not downloaded")

REFUND = ("Refunds to cards and UPI usually arrive within 5 to 7 working days, "
          "while refunds to the Ola wallet arrive within 24 hours.")
FAKE_KEY = "gsk_test_not_a_real_key_0123456789"


def fake_composer(answer: str) -> MockLLM:
    """A stand-in for the Groq composer: like the real model, it writes only the reply text."""
    return MockLLM(responder=lambda task, obs: json.dumps({"answer": answer}))


class FakeJudge:
    def __init__(self, reply):
        self.reply = reply

    def call(self, messages):
        return self.reply


@pytest.fixture
def groq_mode(monkeypatch):
    """Real-model mode with a fake key; the composer is replaced by each test."""
    monkeypatch.setattr(config, "MOCK_LLM", False)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    return monkeypatch


@pytest.fixture(scope="module")
def client():
    from ola_agent.rag.index import build_indexes, get_client

    c = get_client(tempfile.mkdtemp())
    build_indexes(client=c)
    return c


def _ask(groq_mode, client, answer, question="How long does a refund to my card or UPI take?"):
    import ola_agent.crew as crew

    groq_mode.setattr(crew, "_real_llm", lambda: fake_composer(answer))
    from ola_agent.pipeline import ask

    return ask(question, client)


# --- The switch -------------------------------------------------------------------------------

def test_groq_llm_points_at_groq_with_default_model(monkeypatch):
    from ola_agent.real_llm import groq_llm

    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    llm = groq_llm()  # builds the client only; no request is sent
    assert llm.model == "llama-3.3-70b-versatile"
    assert llm.base_url == "https://api.groq.com/openai/v1"
    assert llm.temperature == 0


def test_env_file_gives_only_groq_settings(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={FAKE_KEY}\nGROQ_MODEL=llama-3.1-8b-instant\nMOCK_LLM=false\nOTHER=1\n")
    for name in ("GROQ_API_KEY", "GROQ_MODEL", "MOCK_LLM", "OTHER"):
        monkeypatch.delenv(name, raising=False)
    config._load_groq_settings(str(env))
    assert os.environ["GROQ_API_KEY"] == FAKE_KEY and os.environ["GROQ_MODEL"] == "llama-3.1-8b-instant"
    assert "MOCK_LLM" not in os.environ and "OTHER" not in os.environ


def test_real_composer_is_asked_for_the_reply_text_only(groq_mode):
    # First real Groq run (2026-10-14): asked for the full SupportResponse, the model filled
    # "ticket" with record_id null and Groq rejected the reply against the schema.
    import ola_agent.crew as crew

    groq_mode.setattr(crew, "_real_llm", lambda: fake_composer("x"))
    built, _, _ = crew.build_crew("What priority does a stranded rider get?")
    schema = built.tasks[2].output_pydantic.model_json_schema()
    assert set(schema["properties"]) == {"answer"}
    assert "full sentences" in built.tasks[2].description


def test_mock_mode_never_builds_the_real_model(monkeypatch):
    import ola_agent.crew as crew

    def boom():
        raise AssertionError("real model built in mock mode")

    monkeypatch.setattr(crew, "_real_llm", boom)
    crew.build_crew("What is the status of TKT-0007?")


# --- Sentence check ---------------------------------------------------------------------------

def test_supported_sentences_rules():
    kept, cut = supported_sentences(
        "Refunds to cards and UPI usually arrive within 5 to 7 working days. "
        "Refunds to the Ola wallet arrive within 2 hours. "
        "Refunds to cards and UPI do not arrive within 5 to 7 working days. "
        "Call 98765 43210 about refunds to cards and UPI within 5 to 7 working days. "
        "You will also get a free ride voucher next week.",
        [REFUND],
    )
    assert kept == ["Refunds to cards and UPI usually arrive within 5 to 7 working days."]
    assert len(cut) == 4  # made-up number, added "not", phone number, invented fact


def test_model_answer_kept_in_its_own_words_and_extra_fact_cut():
    chunks = [{"chunk_id": "KB-04-F01", "doc_id": "KB-04", "similarity": 0.7, "text": REFUND}]
    draft = SupportResponse(answer="Card and UPI refunds usually arrive within 5 to 7 working days. "
                                   "Ola also gives every rider a free ride voucher.",
                            sources=["KB-04"], ticket=None, grounded=True, confidence=0.7)
    out = check_groundedness(draft, chunks, 0.433, model_written=True)
    assert out.response.answer == "Card and UPI refunds usually arrive within 5 to 7 working days."
    assert out.fired and "cut 1 sentence" in out.reason and out.response.sources == ["KB-04"]


def test_model_answer_with_nothing_supported_is_refused():
    chunks = [{"chunk_id": "KB-04-F01", "doc_id": "KB-04", "similarity": 0.7, "text": REFUND}]
    draft = SupportResponse(answer="I don't know.", sources=[], ticket=None, grounded=False, confidence=0.7)
    out = check_groundedness(draft, chunks, 0.433, model_written=True)
    assert out.refused and out.response.answer == FALLBACK_ANSWER


def test_mock_mode_rebuilds_from_passages_as_before():
    chunks = [{"chunk_id": "KB-04-F01", "doc_id": "KB-04", "similarity": 0.7, "text": REFUND}]
    draft = SupportResponse(answer="anything", sources=["KB-04"], ticket=None, grounded=True, confidence=0.7)
    assert check_groundedness(draft, chunks, 0.433).response.answer == REFUND


# --- End to end with the fake composer --------------------------------------------------------

@needs_model
def test_pipeline_uses_the_model_answer(groq_mode, client):
    r = _ask(groq_mode, client, "Refunds to cards and UPI usually arrive within 5 to 7 working days. "
                                "They are always instant on weekends.")
    assert r.response.answer == "Refunds to cards and UPI usually arrive within 5 to 7 working days."
    assert r.response.sources == ["KB-04"]
    assert any("cut 1 sentence" in f for f in r.guardrails_fired)


@needs_model
def test_ticket_comes_from_the_tool_not_the_model(groq_mode, client):
    r = _ask(groq_mode, client, "Ticket TKT-0007 is Resolved.", question="What is the status of TKT-0007?")
    from ola_agent.guardrails import ticket_sentence
    from ola_agent.tools.ticket_lookup import check_support_ticket_status

    real = check_support_ticket_status("TKT-0007")
    assert real["status"] != "Resolved"
    assert r.response.ticket.status == real["status"]
    # The model's wrong status is cut, and the tool's own result is given instead.
    assert r.response.answer == ticket_sentence(r.response.ticket)
    assert "Resolved" not in r.response.answer


# --- Judge ------------------------------------------------------------------------------------

def test_judge_reads_a_fenced_json_reply():
    reply = 'Here you go:\n```json\n{"accuracy": 1, "grounding": 0.5, "completeness": 1, "safety": 1, "reason": "ok"}\n```'
    s = judge(TEST_SET[3], REFUND, ["KB-04"], [{"text": REFUND}], FakeJudge(reply))
    assert (s.accuracy, s.grounding) == (1.0, 0.5)


def test_judge_rejects_a_malformed_reply():
    case = EvalCase("Q?", "in_scope", "KB-04", ("5 to 7 working days",))
    for reply in ("All good, full marks.", '{"accuracy": 2, "grounding": 1, "completeness": 1, "safety": 1, "reason": "x"}'):
        with pytest.raises(Exception):
            judge(case, REFUND, ["KB-04"], [], FakeJudge(reply))


# --- The key never leaks ----------------------------------------------------------------------

@needs_model
def test_key_never_in_log_or_response(groq_mode, client):
    from fastapi.testclient import TestClient

    import ola_agent.crew as crew
    from ola_agent.api import app

    log = os.path.join(tempfile.mkdtemp(), "requests.jsonl")
    groq_mode.setattr(request_log, "LOG_PATH", log)
    groq_mode.setattr(crew, "_real_llm", lambda: fake_composer(REFUND))
    app.state.chroma_client = client
    try:
        r = TestClient(app).post("/ask", json={"question": f"my key is {FAKE_KEY}, how long does a refund take?"})
    finally:
        app.state.chroma_client = None
    with open(log, encoding="utf-8") as f:
        text = f.read()
    assert r.status_code == 200
    assert FAKE_KEY not in r.text
    assert not PHONE_PATTERN.search(r.json()["answer"])
    # The question is logged as the user typed it; the server's own key never is.
    assert text.count(FAKE_KEY) == 1
