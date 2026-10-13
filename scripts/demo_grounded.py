"""Grounded generation demo: 5 in-scope queries plus 1 out-of-scope query (Task 4).

Run:  python scripts/demo_grounded.py [collection]   (default: kb_sentence)
"""

import sys

import _bootstrap  # noqa: F401

from ola_agent.rag.generate import FALLBACK_ANSWER, answer, load_threshold
from ola_agent.rag.queries import DEMO_OUT_OF_SCOPE, IN_SCOPE

name = sys.argv[1] if len(sys.argv) > 1 else "kb_sentence"
threshold = load_threshold(name)
print(f"Collection: {name}   threshold: {threshold:.3f}   MOCK_LLM: answers are built only from retrieved chunks\n")

ok = True
for q in IN_SCOPE + [DEMO_OUT_OF_SCOPE]:
    expect_fallback = q == DEMO_OUT_OF_SCOPE
    a = answer(q, name)
    passed = (a.answer == FALLBACK_ANSWER) == expect_fallback
    ok &= passed
    print(f"Q: {q}")
    print(f"   top-1 similarity {a.top_similarity:.3f} -> {'grounded' if a.grounded else 'FALLBACK'}")
    print(f"   sources: {', '.join(a.sources) or '-'}")
    print(f"   A: {a.answer}")
    print(f"   expected {'fallback' if expect_fallback else 'grounded answer'}: {'PASS' if passed else 'FAIL'}\n")
print("GROUNDED DEMO:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
