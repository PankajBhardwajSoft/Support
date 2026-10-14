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


def _snap_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it neither starts nor ends inside a word.

    If the window has no space to snap to (one very long word), it is left as is.
    """
    if start > 0 and not text[start - 1].isspace():
        space = text.find(" ", start, end)
        if space != -1:
            start = space + 1
    if end < len(text) and not text[end].isspace():
        space = text.rfind(" ", start, end)
        if space != -1:
            end = space
    return start, end


def fixed_size_chunks(doc: Document, size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP) -> list[Chunk]:
    """Character windows of at most `size`, each starting `size - overlap` after the last.

    Window edges are pulled in to the nearest space so no word is cut in half.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    step = size - overlap
    chunks = []
    start = 0
    while True:
        lo, hi = _snap_to_words(doc.text, start, min(start + size, len(doc.text)))
        piece = doc.text[lo:hi].strip()
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
