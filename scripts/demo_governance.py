"""Task 15: least autonomy, risk classification and the runtime budget cap.

Uses FastAPI's TestClient in-process for the budget cap, so no port is opened
and no network is needed.
"""

import _bootstrap  # noqa: F401

from fastapi.testclient import TestClient

from ola_agent import request_log
from ola_agent.api import app
from ola_agent.crew import build_crew
from ola_agent.governance import ALLOWED_TOOLS, MAX_REQUEST_TOKENS, check_tool_permissions, estimate_tokens
from ola_agent.pipeline import ask
from ola_agent.tools.ticket_lookup import TicketLookupTool

RISK = ("Risk classification: MEDIUM. The brief's scheme lists customer support tickets under Medium. "
        "The agent reads ticket records and answers policy questions for riders and drivers, so a wrong "
        "answer can mislead a customer about a refund or an SLA. It takes no actions: it cannot issue "
        "refunds, change accounts or close tickets, and escalations stay with a human. Phone numbers are "
        "masked before anything is stored, and every answer is checked against the policy text.")


def main():
    ok = True
    print("1) Least autonomy: which agent may hold which tool")
    for role, tools in ALLOWED_TOOLS.items():
        print(f"   {role}: {sorted(tools) or 'no tools'}")
    crew, _, _ = build_crew("What is the status of TKT-0023?")
    holders = [a.role for a in crew.agents if any(t.name == "check_support_ticket_status" for t in a.tools)]
    print(f"   agents wired with check_support_ticket_status: {holders}")
    ok &= holders == ["Lookup Agent"]

    composer = next(a for a in crew.agents if a.role == "Response Composer")
    composer.tools = [TicketLookupTool()]  # try to give the Composer the ticket tool
    try:
        check_tool_permissions(crew.agents)
        print("   Composer given the ticket tool: NOT blocked (FAIL)")
        ok = False
    except PermissionError as exc:
        print(f"   Composer given the ticket tool: blocked with PermissionError: {exc}")

    r = ask("What is the status of TKT-0023?")
    print(f"   ticket question still answered by the Lookup Agent: calls={r.crew.ticket_tool_calls}")
    print(f"   answer: {r.response.answer}")
    ok &= r.crew.ticket_tool_calls == ["TKT-0023"]

    print(f"\n2) {RISK}")

    print(f"\n3) Budget cap: {MAX_REQUEST_TOKENS} tokens per request (estimate: max(1.3 x words, chars / 4))")
    http = TestClient(app)
    before = len(request_log.read_entries())
    normal = "How long does a refund to my card or UPI take?"
    resp = http.post("/ask", json={"question": normal})
    print(f"   normal question, about {estimate_tokens(normal)} tokens: HTTP {resp.status_code}, "
          f"answer starts: {resp.json()['answer'][:60]}...")
    big = "Please check my refund. " * 400
    resp2 = http.post("/ask", json={"question": big})
    print(f"   oversized simulated request, about {estimate_tokens(big)} tokens: HTTP {resp2.status_code}, "
          f"detail: {resp2.json()['detail']}")
    entry = request_log.read_entries()[before:][-1]
    print(f"   its log line: status={entry['status']}, guardrails_fired={entry['guardrails_fired']}")
    ok &= resp.status_code == 200 and resp2.status_code == 413 and entry["status"] == 413

    print(f"\nGOVERNANCE DEMO: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
