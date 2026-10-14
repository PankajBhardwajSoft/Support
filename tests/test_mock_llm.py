"""Step 1 tests: MockLLM behaviour, the brief's two pitfalls, and offline runs."""

import os
import subprocess
import sys

from crewai import Agent, Crew, Task
from crewai.tools import tool

from ola_agent import netguard
from ola_agent.mock_llm import MockLLM, build_tool_args, extract_observations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The exact template line CrewAI puts in its own system prompt.
CREWAI_TEMPLATE = (
    "You are X.\nIMPORTANT: Use the following format in your response:\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
)


def _user(task: str) -> dict:
    return {"role": "user", "content": f"\nCurrent Task: {task}\n\nThis is the expected criteria for your final answer: x\n\nBegin!"}


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Search the knowledge base."""
    return f"KB:{query}"


def _run_crew(task_text: str, tools: list, llm: MockLLM | None = None) -> tuple[str, MockLLM]:
    llm = llm or MockLLM()
    agent = Agent(role="R", goal="g", backstory="b", tools=tools, llm=llm, verbose=False)
    task = Task(description=task_text, expected_output="answer", agent=agent)
    return Crew(agents=[agent], tasks=[task]).kickoff().raw, llm


# Pitfall 1: template "Observation:" text must not be read as a tool result.
def test_template_observation_is_not_a_result():
    msgs = [{"role": "system", "content": CREWAI_TEMPLATE}, _user("What is the refund policy?")]
    assert extract_observations(msgs) == []


def test_first_call_uses_tool_instead_of_returning_template_text():
    out, _ = _run_crew("What is the refund policy?", [rag_lookup])
    assert out == "KB:What is the refund policy?"
    assert "the result of the action" not in out


# Pitfall 2: arguments come from the schema, not the tool's name.
def test_rag_lookup_name_does_not_make_it_a_ticket_tool():
    schema = rag_lookup.args_schema.model_json_schema()
    assert build_tool_args(schema, "Status of TKT-0007?") == {"query": "Status of TKT-0007?"}


def test_record_id_schema_is_filled_whatever_the_tool_is_called():
    @tool("rag_lookup_misleading_name")
    def misleading(record_id: str) -> str:
        """Ticket lookup with a confusing name."""
        return f"TICKET:{record_id}"

    out, _ = _run_crew("Check status of TKT-0007 please", [misleading])
    assert out == "TICKET:TKT-0007"


def test_tool_skipped_when_required_field_cannot_be_filled():
    schema = {"properties": {"record_id": {"type": "string"}}, "required": ["record_id"]}
    assert build_tool_args(schema, "What is the refund policy?") is None


# Determinism and offline.
def test_same_input_gives_same_output():
    msgs = [{"role": "system", "content": CREWAI_TEMPLATE}, _user("Hello")]
    assert MockLLM().call(msgs) == MockLLM().call(msgs)


def test_crew_makes_no_network_attempts():
    netguard.blocked_attempts.clear()
    _run_crew("What is the refund policy?", [rag_lookup])
    assert netguard.blocked_attempts == []


def test_call_counter_counts_llm_calls():
    _, llm = _run_crew("What is the refund policy?", [rag_lookup])
    assert llm.call_count == 2  # one tool call, one final answer


# Config.
def _python(code: str, env_extra: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_extra}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)


def test_telemetry_forced_off_even_if_shell_enables_it():
    r = _python(
        "import os, ola_agent; print(os.environ['CREWAI_DISABLE_TELEMETRY'], os.environ['OTEL_SDK_DISABLED'])",
        {"CREWAI_DISABLE_TELEMETRY": "false", "OTEL_SDK_DISABLED": "false"},
    )
    assert r.stdout.strip() == "true true"


def test_mock_llm_false_is_refused():
    r = _python("import ola_agent", {"MOCK_LLM": "false"})
    assert r.returncode != 0
    assert "MOCK_LLM=false is not supported" in r.stderr
