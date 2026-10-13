"""The two chunking strategies the brief requires (Part 1, Task 3)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ola_agent.rag.kb import Document

FIXED_CHUNK_SIZE = 200
FIXED_CHUNK_OVERLAP = 50

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def fixed_size_chunks(doc: Document, size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP) -> list[Chunk]:
    """Character windows of `size`, each starting `size - overlap` after the last."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    step = size - overlap
    chunks = []
    start = 0
    while True:
        piece = doc.text[start:start + size].strip()
        if piece:
            chunks.append(Chunk(f"{doc.doc_id}-F{len(chunks):02d}", doc.doc_id, piece))
        if start + size >= len(doc.text):
            break
        start += step
    return chunks


def sentence_chunks(doc: Document) -> list[Chunk]:
    """One chunk per sentence (split after . ! or ? followed by whitespace)."""
    sentences = [s.strip() for s in _SENTENCE_END.split(doc.text) if s.strip()]
    return [Chunk(f"{doc.doc_id}-S{i:02d}", doc.doc_id, s) for i, s in enumerate(sentences)]


STRATEGIES = {
    "kb_fixed": fixed_size_chunks,
    "kb_sentence": sentence_chunks,
}
