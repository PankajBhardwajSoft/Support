"""Task 8: multi-turn memory in one session, then a fresh session with no memory."""

import _bootstrap  # noqa: F401

from ola_agent.memory import ASK_FOR_TICKET, ChatSessions, get_history, reset_sessions


def turn(sessions, sid, q):
    out = sessions.chat(sid, q)
    print(f"[{sid}] user: {q}")
    if out["resolved_question"] != q:
        print(f"[{sid}] (memory supplied the ticket: {out['resolved_question']!r})")
    print(f"[{sid}] agent: {out['answer']}")
    return out


def main():
    reset_sessions()
    sessions = ChatSessions()
    print("=== Transcript 1: one session, state carries across turns ===")
    turn(sessions, "session-1", "What is the status of TKT-0007?")
    first = turn(sessions, "session-1", "How long will it take to resolve?")
    print(f"history now holds {len(get_history('session-1').messages)} messages\n")

    print("=== Transcript 2: fresh session, state is absent ===")
    fresh = turn(sessions, "session-2", "How long will it take to resolve?")
    print(f"history now holds {len(get_history('session-2').messages)} messages\n")

    carried = first["response"].ticket is not None and first["response"].ticket.record_id == "TKT-0007"
    absent = fresh["answer"] == ASK_FOR_TICKET
    print("Turn 2 used TKT-0007 from memory:", "PASS" if carried else "FAIL")
    print("Fresh session had no ticket in memory:", "PASS" if absent else "FAIL")
    print("MEMORY DEMO:", "PASS" if carried and absent else "FAIL")
    return 0 if carried and absent else 1


if __name__ == "__main__":
    raise SystemExit(main())
