"""Measure top-1 cosine similarity for in-scope and out-of-scope queries and set
the "I don't know" threshold between the two observed clusters (Task 4).

Run:  python scripts/calibrate_threshold.py
Writes rag_thresholds.json.
"""

import json

import _bootstrap  # noqa: F401

from ola_agent.rag.generate import THRESHOLDS_FILE, retrieve
from ola_agent.rag.queries import IN_SCOPE, OUT_OF_SCOPE

results = {}
ok = True
for name in ("kb_fixed", "kb_sentence"):
    print(f"Collection '{name}': top-1 cosine similarity")
    ins = [(q, retrieve(q, name, k=1)[0].similarity) for q in IN_SCOPE]
    outs = [(q, retrieve(q, name, k=1)[0].similarity) for q in OUT_OF_SCOPE]
    for q, s in ins:
        print(f"  in-scope      {s:.3f}  {q}")
    for q, s in outs:
        print(f"  out-of-scope  {s:.3f}  {q}")
    lo_in = min(s for _, s in ins)
    hi_out = max(s for _, s in outs)
    if lo_in <= hi_out:
        print(f"  CLUSTERS OVERLAP: lowest in-scope {lo_in:.3f} <= highest out-of-scope {hi_out:.3f}. No threshold set.\n")
        ok = False
        continue
    threshold = round((lo_in + hi_out) / 2, 3)
    print(f"  lowest in-scope = {lo_in:.3f}, highest out-of-scope = {hi_out:.3f}, gap = {lo_in - hi_out:.3f}")
    print(f"  threshold = midpoint = ({lo_in:.3f} + {hi_out:.3f}) / 2 = {threshold:.3f}\n")
    results[name] = {
        "threshold": threshold,
        "lowest_in_scope": round(lo_in, 3),
        "highest_out_of_scope": round(hi_out, 3),
        "in_scope": {q: round(s, 3) for q, s in ins},
        "out_of_scope": {q: round(s, 3) for q, s in outs},
    }

if ok:
    with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {THRESHOLDS_FILE.rsplit('/', 1)[-1]}")
print("CALIBRATION:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
