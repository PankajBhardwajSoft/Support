"""Step 2 tests that need no embedding model: dataset, KB loading, chunking."""

import re

import dataset
from ola_agent.rag.chunking import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE, fixed_size_chunks, sentence_chunks
from ola_agent.rag.kb import Document, load_documents

REQUIRED_TOPICS = [
    "priority", "sla", "escalation", "refund", "communication", "business-hours",
    "repeat-complaint", "service-credit", "feedback", "vip", "outage", "retention",
]


def test_dataset_meets_every_threshold():
    assert all(ok for _, ok in dataset.validate(dataset.SUPPORT_TICKETS))


def test_dataset_fields_and_ranges():
    for t in dataset.SUPPORT_TICKETS:
        assert set(t) == {"record_id", "category", "status", "resolution_time_hours", "days_since_created", "escalated"}
        assert re.fullmatch(r"TKT-\d{4}", t["record_id"])
        assert 1.0 <= t["resolution_time_hours"] <= 168.0
        assert isinstance(t["days_since_created"], int) and 0 <= t["days_since_created"] <= 30
        assert isinstance(t["escalated"], bool)
        if t["status"] == "Escalated":
            assert t["escalated"] is True


def test_dataset_is_deterministic():
    assert dataset.generate_tickets(42) == dataset.generate_tickets(42)
    assert dataset.generate_tickets(42) != dataset.generate_tickets(43)


def test_kb_has_12_docs_with_2_to_5_sentences():
    docs = load_documents()
    assert len(docs) >= 12
    assert len({d.doc_id for d in docs}) == len(docs)
    for d in docs:
        assert 2 <= len(sentence_chunks(d)) <= 5, d.doc_id


def test_kb_covers_every_required_topic():
    titles = " ".join(d.title.lower().replace(" ", "-") for d in load_documents())
    for topic in REQUIRED_TOPICS:
        assert topic in titles, topic


def test_fixed_chunks_have_size_and_overlap():
    doc = Document("X", "t", "abcdefghij" * 50)
    chunks = fixed_size_chunks(doc)
    assert all(len(c.text) <= FIXED_CHUNK_SIZE for c in chunks)
    step = FIXED_CHUNK_SIZE - FIXED_CHUNK_OVERLAP
    assert chunks[0].text[step:] == chunks[1].text[:FIXED_CHUNK_OVERLAP]
    assert "".join(c.text[: step] for c in chunks[:-1]) + chunks[-1].text == doc.text


def test_every_chunk_keeps_its_parent_doc():
    for d in load_documents():
        for c in fixed_size_chunks(d) + sentence_chunks(d):
            assert c.doc_id == d.doc_id and c.chunk_id.startswith(d.doc_id)
