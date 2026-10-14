"""Project settings, read from environment variables.

This module must be imported before `crewai` so CrewAI's telemetry is off
before any crew runs (the brief's zero-network rule).
"""

import os

# CrewAI sends telemetry on crew.kickoff() unless these are set.
# Forced on (not setdefault) so a stray shell variable cannot re-enable it.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Deterministic mock LLM, zero API keys, zero network. Graded runs use this.
MOCK_LLM: bool = _env_flag("MOCK_LLM", True)

if not MOCK_LLM:
    # A real LLM is optional in the brief and not implemented (PRD decision D18).
    raise RuntimeError(
        "MOCK_LLM=false is not supported in this project. "
        "Unset MOCK_LLM or set it to true."
    )
