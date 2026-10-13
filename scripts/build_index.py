"""Chunk the KB both ways and index each strategy in its own ChromaDB collection (Task 3).

Run:  python scripts/build_index.py
"""

import _bootstrap  # noqa: F401

from ola_agent.rag.chunking import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE
from ola_agent.rag.generate import retrieve
from ola_agent.rag.index import build_indexes
from ola_agent.rag.kb import load_documents

docs = load_documents()
print(f"Loaded {len(docs)} KB documents: {', '.join(d.doc_id for d in docs)}")
print(f"Fixed-size chunks: {FIXED_CHUNK_SIZE} chars, {FIXED_CHUNK_OVERLAP} overlap. Sentence chunks: one per sentence.")
for name, n in build_indexes(docs).items():
    print(f"Upserted {n} chunks into collection '{name}'")

sample = "What is the refund policy for a double charge?"
print(f"\nSample query: {sample!r}")
for name in ("kb_fixed", "kb_sentence"):
    print(f"  [{name}]")
    for r in retrieve(sample, name):
        print(f"    {r.similarity:.3f}  {r.chunk_id}  {r.text[:80]}")
