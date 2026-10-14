"""Step 1 smoke test: a one-agent CrewAI crew runs on MockLLM with the network blocked.

Run:  python scripts/smoke_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ola_agent import config  # noqa: E402  (sets telemetry off before crewai)
from ola_agent.netguard import block_network, blocked_attempts  # noqa: E402

block_network()

from crewai import Agent, Crew, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

from ola_agent.mock_llm import MockLLM  # noqa: E402


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Placeholder knowledge-base search (the real one is built in step 2)."""
    return f"[placeholder policy text for: {query}]"


def main() -> None:
    print("MOCK_LLM =", config.MOCK_LLM)
    print("CREWAI_DISABLE_TELEMETRY =", os.environ["CREWAI_DISABLE_TELEMETRY"])
    print("OTEL_SDK_DISABLED =", os.environ["OTEL_SDK_DISABLED"])
    print("Network: blocked for this run")

    llm = MockLLM()
    agent = Agent(
        role="Smoke Test Agent",
        goal="Prove the mock LLM and tool loop work offline",
        backstory="A temporary agent used only for the step 1 smoke test.",
        tools=[rag_lookup],
        llm=llm,
        verbose=False,
    )
    task = Task(
        description="What is the refund policy?",
        expected_output="The placeholder policy text returned by the tool.",
        agent=agent,
    )
    result = Crew(agents=[agent], tasks=[task]).kickoff()

    print("Crew answer:", result.raw)
    print("Mock LLM calls:", llm.call_count)
    print("Blocked network attempts:", blocked_attempts or "none")
    ok = not blocked_attempts and result.raw == "[placeholder policy text for: What is the refund policy?]"
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
