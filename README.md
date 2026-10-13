# Ola Domain Support Agent (CrewAI)

**Track completed: Ola (Business Operations / Customer Support).**

A support agent that answers Ola support-policy questions from a knowledge base and checks support-ticket status from a generated dataset. Built with CrewAI, an Autogen review stage, LangChain session memory, ChromaDB and SentenceTransformers, behind a FastAPI backend.

> Status: step 1 of 6 (base setup). See `TRACKER.md` for progress against every requirement.

## Runs offline with zero API keys

- Every language-model call uses a deterministic mock (`MOCK_LLM`, on by default). No API key and no network are needed.
- The mock extends CrewAI's own extension point, `crewai.llms.base_llm.BaseLLM` (`ola_agent/mock_llm.py`).
- **CrewAI telemetry is disabled.** `ola_agent/config.py` sets `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` before CrewAI is imported, and overrides any other value already set in the shell.
- Setting `MOCK_LLM=false` stops the program with an error; a real LLM is not wired in.

## Dataset design choices

Filled in at step 2 (seed, category/status weights, ranges).

## Setup in VS Code (Windows)

1. Install Python 3.11 and VS Code with the Python extension.
2. Open this folder in VS Code, then open a terminal (**Terminal > New Terminal**) and run:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

3. Select the interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**.
4. Run from the **Run and Debug** panel:
   - **Step 1: smoke test**
   - **All tests (pytest)**

   Or from the terminal:

   ```powershell
   python scripts\smoke_test.py
   python -m pytest -q
   ```

## Project layout

```
ola_agent/
  config.py       settings; turns telemetry off; MOCK_LLM flag
  mock_llm.py     deterministic offline LLM for CrewAI (BaseLLM)
  netguard.py     blocks outbound network during runs and tests
scripts/
  smoke_test.py   step 1: one-agent crew on the mock, network blocked
tests/            unit tests and attack tests (all run with network blocked)
transcripts/      saved text output of every demonstration run
security/         attack-test reports, one per step
TRACKER.md        every requirement mapped to code and proof
```

## How the mock avoids the brief's two silent pitfalls

1. CrewAI's system prompt contains the example line `Observation: the result of the action`. The mock reads tool results **only from assistant-role messages** (its own turns, where CrewAI appends the real tool output), never from system or user messages. Test: `test_template_observation_is_not_a_result`.
2. Tool arguments are filled from each tool's **declared argument schema** (field names such as `record_id` or `query`), never from the tool's name. A tool named `rag_lookup` is never treated as the ticket lookup. Tests: `test_rag_lookup_name_does_not_make_it_a_ticket_tool`, `test_record_id_schema_is_filled_whatever_the_tool_is_called`.
