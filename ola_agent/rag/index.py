"""Embeds chunks and indexes each chunking strategy in its own ChromaDB collection."""

from __future__ import annotations

import os

import chromadb
from chromadb.config import Settings

from ola_agent.rag.chunking import STRATEGIES, Chunk
from ola_agent.rag.embeddings import ROOT, embed
from ola_agent.rag.kb import Document, load_documents

CHROMA_DIR = os.path.join(ROOT, "chroma_db")


def get_client(path: str = CHROMA_DIR):
    # anonymized_telemetry=False keeps ChromaDB from sending usage data.
    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))


def get_collection(name: str, client=None):
    client = client or get_client()
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(collection, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"doc_id": c.doc_id} for c in chunks],
        embeddings=embed([c.text for c in chunks]),
    )


def build_indexes(docs: list[Document] | None = None, client=None) -> dict[str, int]:
    """Chunk every document with both strategies and upsert into two collections."""
    docs = docs if docs is not None else load_documents()
    client = client or get_client()
    counts = {}
    for name, strategy in STRATEGIES.items():
        chunks = [c for d in docs for c in strategy(d)]
        upsert_chunks(get_collection(name, client), chunks)
        counts[name] = len(chunks)
    return counts
