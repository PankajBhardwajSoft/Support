"""Tasks 11 and 12: call the FastAPI app in-process and show the JSON-Lines log it writes.

Uses FastAPI's TestClient, so no port is opened and no network is needed.
A temporary copy of the index is used, so the document added here does not
change ./chroma_db.
"""

import json
import os
import tempfile

import _bootstrap  # noqa: F401

from fastapi.testclient import TestClient

from ola_agent import request_log
from ola_agent.api import app
from ola_agent.guardrails import PHONE_PATTERN
from ola_agent.memory import ASK_FOR_TICKET
from ola_agent.rag.embeddings import ROOT
from ola_agent.rag.index import build_indexes, get_client

NEW_DOC = {
    "doc_id": "KB-13",
    "title": "Lost item policy",
    "text": "If a rider leaves an item in the cab, support connects them with the driver through in-app chat "
            "within 2 hours. A lost item return fee of 100 rupees is added to the rider's next trip.",
}


def show(label, value):
    print(f"{label}: {json.dumps(value)}")


def main():
    client = get_client(tempfile.mkdtemp())
    build_indexes(client=client)
    app.state.chroma_client = client
    before = len(request_log.read_entries())
    http = TestClient(app)
    checks = {}
    traces = []

    print("== POST /ask (policy question)")
    r = http.post("/ask", json={"question": "When does a ticket get escalated to Tier 2?"})
    show(f"HTTP {r.status_code}, X-Trace-ID {r.headers['x-trace-id']}", r.json())
    traces.append(r.json()["trace_id"])
    checks["/ask answers a policy question from KB-03"] = r.status_code == 200 and r.json()["sources"][:1] == ["KB-03"]

    print("\n== POST /ask (ticket question with a phone number in it)")
    r = http.post("/ask", json={"question": "Call me on +91 98765 43210. What is the status of TKT-0023?"})
    show(f"HTTP {r.status_code}", r.json())
    traces.append(r.json()["trace_id"])
    checks["/ask looks up TKT-0023"] = r.status_code == 200 and "TKT-0023" in r.json()["answer"]

    print("\n== POST /add-document, then /ask finds the new document")
    r = http.post("/add-document", json=NEW_DOC)
    show(f"HTTP {r.status_code}", r.json())
    traces.append(r.json()["trace_id"])
    r2 = http.post("/ask", json={"question": "What happens if I leave an item in the cab?"})
    show(f"HTTP {r2.status_code}", r2.json())
    traces.append(r2.json()["trace_id"])
    checks["/add-document adds KB-13 and /ask uses it"] = (
        r.status_code == 200 and r.json()["chunks_added"] > 0 and "KB-13" in r2.json()["sources"])
    r3 = http.post("/add-document", json={**NEW_DOC, "doc_id": "KB-03"})
    show(f"Overwrite KB-03 -> HTTP {r3.status_code}", r3.json())
    checks["an existing document cannot be overwritten"] = r3.status_code == 409

    print("\n== WebSocket /ws/chat: client A disconnects mid-answer, client B keeps chatting")
    with http.websocket_connect("/ws/chat") as b:
        with http.websocket_connect("/ws/chat") as a:
            a.send_text("What is the status of TKT-0007?")
            print("A sent a question and disconnected before reading the answer")
        b.send_text("What is the status of TKT-0023?")
        b1 = b.receive_json()
        show("B turn 1", b1)
        b.send_text("How long will it take to resolve it?")
        b2 = b.receive_json()
        show("B turn 2", b2)
        traces += [b1["trace_id"], b2["trace_id"]]
    checks["B is still served after A disconnects"] = "TKT-0023" in b1["answer"]
    checks["B's follow-up uses TKT-0023 from memory"] = "TKT-0023" in b2["answer"]
    with http.websocket_connect("/ws/chat") as c:
        c.send_text("What is the status of my ticket?")
        c1 = c.receive_json()
        show("New client C", c1)
        traces.append(c1["trace_id"])
    checks["a new client gets no other client's ticket"] = c1["answer"] == ASK_FOR_TICKET

    print(f"\n== New lines in {os.path.relpath(request_log.LOG_PATH, ROOT)}")
    new = request_log.read_entries()[before:]
    for e in new:
        print(json.dumps(e))
    logged = [e["trace_id"] for e in new]
    checks["every request and message has exactly one log line"] = all(logged.count(t) == 1 for t in traces)
    checks["every log line has a trace ID and timing or status"] = all(e["trace_id"] and e["status"] for e in new)
    # Trace and session IDs are random hex and are skipped: they can contain digit runs by chance.
    text_fields = [json.dumps({k: v for k, v in e.items() if k not in ("trace_id", "session_id", "ts")}) for e in new]
    checks["no raw phone number in the log"] = not any(PHONE_PATTERN.search(t) for t in text_fields)

    print()
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    ok = all(checks.values())
    print("API DEMO:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
