"""Grounded generation with an empirically calibrated "I don't know" fallback (Task 4).

Under MOCK_LLM there is no model to judge groundedness, so retrieval
similarity is the only signal: if the best chunk's cosine similarity is below
the calibrated threshold, the answer is "I don't know". Otherwise the answer
is built only from the retrieved chunk text, never from anything else.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ola_agent.rag.embeddings import ROOT, embed
from ola_agent.rag.index import get_collection

TOP_K = 3
# Chosen from scripts/eval_retrieval.py: higher mean precision at equal recall (README).
RECOMMENDED_COLLECTION = "kb_fixed"
FALLBACK_ANSWER = "I don't know. I couldn't find this in the Ola support policies."

# Written by scripts/calibrate_threshold.py from measured similarities; see README.
THRESHOLDS_FILE = os.path.join(ROOT, "rag_thresholds.json")


def load_threshold(collection_name: str) -> float:
    if not os.path.exists(THRESHOLDS_FILE):
        raise RuntimeError("No calibrated threshold yet; run scripts/calibrate_threshold.py first")
    with open(THRESHOLDS_FILE, encoding="utf-8") as f:
        return float(json.load(f)[collection_name]["threshold"])


@dataclass
class Retrieved:
    chunk_id: str
    doc_id: str
    text: str
    similarity: float


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    grounded: bool
    sources: list[str] = field(default_factory=list)
    top_similarity: float = 0.0
    chunks: list[Retrieved] = field(default_factory=list)


def retrieve(query: str, collection_name: str, k: int = TOP_K, client=None) -> list[Retrieved]:
    collection = get_collection(collection_name, client)
    res = collection.query(query_embeddings=embed([query]), n_results=k)
    return [
        Retrieved(cid, meta["doc_id"], text, 1.0 - dist)  # cosine distance -> similarity
        for cid, meta, text, dist in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0])
    ]


def answer(query: str, collection_name: str, threshold: float | None = None, k: int = TOP_K, client=None) -> GroundedAnswer:
    threshold = load_threshold(collection_name) if threshold is None else threshold
    if not query or not query.strip():
        return GroundedAnswer(query, FALLBACK_ANSWER, grounded=False)
    chunks = retrieve(query, collection_name, k, client)
    top = chunks[0].similarity if chunks else 0.0
    if top < threshold:
        return GroundedAnswer(query, FALLBACK_ANSWER, grounded=False, top_similarity=top, chunks=chunks)
    kept = [c for c in chunks if c.similarity >= threshold]
    sources = list(dict.fromkeys(c.doc_id for c in kept))
    text = " ".join(dict.fromkeys(c.text for c in kept))  # MOCK_LLM: extractive, context only
    return GroundedAnswer(query, text, grounded=True, sources=sources, top_similarity=top, chunks=chunks)
