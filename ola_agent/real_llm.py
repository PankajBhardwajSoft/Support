"""The optional real LLM: Groq through CrewAI's built-in OpenAI client.

Used only when MOCK_LLM=false (see config.py). No extra package is needed:
Groq serves an OpenAI-compatible API.
"""

from __future__ import annotations

import os

from ola_agent import config


def groq_llm():
    from crewai import LLM

    return LLM(
        model=os.environ.get("GROQ_MODEL") or config.GROQ_DEFAULT_MODEL,
        provider="openai",
        base_url=config.GROQ_BASE_URL,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )
