"""Task 7 and 9: run the 3-agent crew with .kickoff() and show which tool each question used."""

import _bootstrap  # noqa: F401

from ola_agent.pipeline import ask

QUESTIONS = [
    "When does a ticket get escalated to Tier 2?",  # policy question: the RAG tool
    "What is the status of TKT-0023?",  # ticket question: the lookup tool
]


def main():
    results = []
    for q in QUESTIONS:
        r = ask(q)
        results.append(r)
        run = r.crew
        print(f"Q: {q}")
        print(f"   agents: {', '.join(run.agents)}")
        print(f"   Retrieval Agent called search_ola_policies: {run.policy_tool_calls}")
        print(f"   Lookup Agent called check_support_ticket_status: {run.ticket_tool_calls}")
        print(f"   Composer draft (validated as SupportResponse): {run.draft.model_dump_json()}")
        print(f"   guardrails: {r.guardrails_fired or 'none fired'}")
        print(f"   final answer: {r.response.model_dump_json(indent=None)}")
        print()
    policy, ticket = results[0].crew, results[1].crew
    used_rag = bool(policy.policy_tool_calls) and not policy.ticket_tool_calls
    used_lookup = ticket.ticket_tool_calls == ["TKT-0023"]
    print("RAG tool used on the policy question only:", "PASS" if used_rag else "FAIL")
    print("Lookup tool used on the ticket question:", "PASS" if used_lookup else "FAIL")
    ok = used_rag and used_lookup
    print("CREW DEMO:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
