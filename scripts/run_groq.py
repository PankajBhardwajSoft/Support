"""Optional real-LLM run: the 15 evaluation questions answered by the Groq composer,
scored by both the rule-based judge and the Groq judge.

Needs MOCK_LLM=false and GROQ_API_KEY in the project's .env file. Unlike the
other scripts, this one does not block the network: it has to reach Groq.
CrewAI telemetry stays off (config.py). Graded transcripts never use this script.

PowerShell:  $env:MOCK_LLM="false"; python scripts/run_groq.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ola_agent import config  # noqa: E402  (telemetry off, key loaded, before crewai)


def main():
    if config.MOCK_LLM:
        print("This script needs the real model. In PowerShell run:  $env:MOCK_LLM=\"false\"")
        return 2

    from ola_agent.evaluation import averages, evaluate, judge
    from ola_agent.real_llm import groq_llm

    model = os.environ.get("GROQ_MODEL") or config.GROQ_DEFAULT_MODEL
    print(f"Groq model: {model}   (composer and judge; retrieval and lookup agents stay on the mock)\n")

    rows = evaluate()  # answers from the Groq composer, scored by the rule-based judge
    llm = groq_llm()
    groq_scores, invalid = [], 0
    for r in rows:
        try:
            groq_scores.append(judge(r.case, r.answer, r.sources, r.chunks, llm))
        except Exception as e:  # a malformed verdict is reported, never scored as a pass
            invalid += 1
            groq_scores.append(None)
            print(f"Groq judge gave no valid verdict for: {r.case.question} ({type(e).__name__})")

    print(f"{'#':>2}  {'kind':<12} {'source':<7} {'rule A/G/C/S':<20} {'Groq A/G/C/S':<20} question")
    for i, (r, g) in enumerate(zip(rows, groq_scores), 1):
        s = r.scores
        rule = f"{s.accuracy:.2f} {s.grounding:.2f} {s.completeness:.2f} {s.safety:.2f}"
        groq = f"{g.accuracy:.2f} {g.grounding:.2f} {g.completeness:.2f} {g.safety:.2f}" if g else "invalid"
        print(f"{i:>2}  {r.case.kind:<12} {(r.sources or ['-'])[0]:<7} {rule:<20} {groq:<20} {r.case.question}")
        print(f"    answer: {r.answer}")

    avg = averages(rows)
    print("\nRule-based judge averages:", avg)
    valid = [g for g in groq_scores if g]
    if valid:
        keys = ("accuracy", "grounding", "completeness", "safety")
        print("Groq judge averages:      ", {k: round(sum(getattr(g, k) for g in valid) / len(valid), 3) for k in keys})
    print(f"Groq verdicts: {len(valid)} valid, {invalid} invalid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
