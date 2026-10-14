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


def _load_groq_settings(path: str | None = None) -> None:
    """Read only the GROQ_* lines of the project's .env file (never committed).

    MOCK_LLM itself is never taken from .env, so a stray file cannot switch
    graded runs or tests to the real model.
    """
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(path):
        return
    from dotenv import dotenv_values

    for name, value in dotenv_values(path).items():
        if name in {"GROQ_API_KEY", "GROQ_MODEL"} and value:
            os.environ.setdefault(name, value)


# Deterministic mock LLM, zero API keys, zero network. Graded runs use this.
MOCK_LLM: bool = _env_flag("MOCK_LLM", True)

# Optional real LLM through Groq (change request approved 2026-10-14, replacing
# PRD decision D18). Only the Response Composer and the eval judge use it.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"

if not MOCK_LLM:
    _load_groq_settings()
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "MOCK_LLM=false needs a Groq API key. Put GROQ_API_KEY=... in the "
            "project's .env file, or unset MOCK_LLM to use the mock."
        )
