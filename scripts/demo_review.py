"""Task 14: the Autogen review stage approves one answer unchanged and revises another.

Both runs use the two-agent RoundRobinGroupChat in ola_agent/review.py on the
mock model client: no key, no network. The second draft has a made-up
sentence planted in it on purpose, as the brief suggests.
"""

import _bootstrap  # noqa: F401

from ola_agent.pipeline import ask
from ola_agent.rag.generate import RECOMMENDED_COLLECTION, load_threshold
from ola_agent.review import EDITOR, REVIEWER, build_team, review

PLANTED = "Ola also refunds the full fare for every late ride."


def evidence_of(result):
    sources, threshold = set(result.response.sources), load_threshold(RECOMMENDED_COLLECTION)
    return [c["text"] for c in result.crew.search["chunks"] if c["doc_id"] in sources and c["similarity"] >= threshold]


def main():
    build_team()  # builds without error: custom_message_types carries the editor's StructuredMessage
    print(f"Team: RoundRobinGroupChat([{REVIEWER}, {EDITOR}]), max_turns=2, "
          "custom_message_types=[StructuredMessage[Verdict]]")
    print(f"{EDITOR} output_content_type: Verdict(approved: bool, final_answer: str, reason: str)\n")

    result = ask("What service credit do I get if my driver arrives very late?")
    evidence = evidence_of(result)
    draft = result.response.answer

    print("1) Real pipeline answer, reviewed inside ask():")
    print(f"   draft: {draft}")
    print(f"   verdict: {result.verdict.model_dump_json()}")
    ok1 = result.verdict.approved and result.verdict.final_answer == draft
    print(f"   approved unchanged: {'PASS' if ok1 else 'FAIL'}\n")

    planted = f"{PLANTED} {draft}"  # first, so it is a sentence of its own
    verdict = review(planted, evidence)
    print("2) The same draft with a made-up sentence planted in it:")
    print(f"   draft: {planted}")
    print(f"   verdict: {verdict.model_dump_json()}")
    ok2 = (not verdict.approved) and PLANTED not in verdict.final_answer and verdict.final_answer == draft
    print(f"   revised, planted sentence removed, nothing added: {'PASS' if ok2 else 'FAIL'}\n")

    print(f"{REVIEWER} and {EDITOR} both run on MockChatClient (no key, no network).")
    print(f"REVIEW DEMO: {'PASS' if ok1 and ok2 else 'FAIL'}")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
