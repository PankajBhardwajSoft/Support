"""Document-level precision and recall for both chunking strategies (Task 5).

Retrieved chunks are mapped back to their parent document and de-duplicated
before scoring. Same 5 in-scope queries as Task 4.

Run:  python scripts/eval_retrieval.py
"""

import _bootstrap  # noqa: F401

from ola_agent.rag.generate import TOP_K, retrieve
from ola_agent.rag.queries import IN_SCOPE, RELEVANT_DOCS

summary = {}
for name in ("kb_fixed", "kb_sentence"):
    print(f"=== Collection '{name}' (top-{TOP_K} chunks -> parent docs, deduplicated) ===")
    ps, rs = [], []
    for q in IN_SCOPE:
        chunks = retrieve(q, name)
        docs = list(dict.fromkeys(c.doc_id for c in chunks))
        relevant = RELEVANT_DOCS[q]
        hits = [d for d in docs if d in relevant]
        p = len(hits) / len(docs)
        r = len(hits) / len(relevant)
        ps.append(p)
        rs.append(r)
        print(f"Q: {q}")
        print(f"   chunks: {', '.join(c.chunk_id for c in chunks)}")
        print(f"   retrieved docs: {docs}   relevant: {sorted(relevant)}   hits: {hits}")
        print(f"   precision = {len(hits)}/{len(docs)} = {p:.3f}   recall = {len(hits)}/{len(relevant)} = {r:.3f}")
    mp, mr = sum(ps) / len(ps), sum(rs) / len(rs)
    summary[name] = (mp, mr)
    print(f"   mean precision = ({' + '.join(f'{x:.3f}' for x in ps)}) / {len(ps)} = {mp:.3f}")
    print(f"   mean recall    = ({' + '.join(f'{x:.3f}' for x in rs)}) / {len(rs)} = {mr:.3f}\n")

print("Summary:")
for name, (mp, mr) in summary.items():
    print(f"  {name:<12} mean precision {mp:.3f}   mean recall {mr:.3f}")
