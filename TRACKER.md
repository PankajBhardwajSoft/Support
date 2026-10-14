# Requirement tracker

Maps every requirement in the capstone brief to the proof in this repo (rule.md section 3c).
A row is **Done** only when its transcript shows it passing under `MOCK_LLM` with the network blocked.

Status values: Not started / In progress / Done / Failed

## Submission guidelines

| # | Requirement | Status | Code file | Transcript | Attack test | Notes |
|---|---|---|---|---|---|---|
| G1 | All LLM calls default to deterministic `MOCK_LLM`, zero API keys, zero network | In progress | `ola_agent/config.py`, `ola_agent/mock_llm.py` | `transcripts/step1_smoke_test.txt` | `security/step1_attack_test.md` | Waiting on finding F1 (network guard lets localhost proxy traffic through) |
| G2 | CrewAI mock extends `crewai.llms.base_llm.BaseLLM` | Done | `ola_agent/mock_llm.py` | `transcripts/step1_pytest.txt` | A1, A7 | |
| G3 | Pitfall 1: template "Observation:" text is never read as a tool result | Done | `ola_agent/mock_llm.py` (`extract_observations`) | `transcripts/step1_pytest.txt` | A1 | |
| G4 | Pitfall 2: tool arguments dispatched by schema, not tool name | Done | `ola_agent/mock_llm.py` (`build_tool_args`) | `transcripts/step1_pytest.txt` | A2, A3 | |
| G5 | `CREWAI_DISABLE_TELEMETRY=true` set before running the crew, confirmed in README | In progress | `ola_agent/config.py` | `transcripts/step1_smoke_test.txt` | A8, A9 | Waiting on finding F2 (negative-control test is invalid) |
| G6 | README states Ola track and dataset design choices | Done | `README.md` | `transcripts/step2_dataset.txt` | B1 | |
| G7 | Everything is code or text; no images, PDFs, slides, video, audio | In progress | whole repo | | pre-push scan | Checked at every push |
| G8 | One public GitHub repository | In progress | github.com/PankajBhardwajSoft/Support | | pre-push scan | Repo exists; final submission at the end |

## Part 1: Dataset design and RAG core

| # | Acceptance criterion | Status | Code file | Transcript | Attack test | Notes |
|---|---|---|---|---|---|---|
| 1.1 | `dataset.py` generates >=40 tickets meeting every threshold; design choices in README | Done | `dataset.py` | `transcripts/step2_dataset.txt` | B1 | 50 tickets, seed 42, escalated 26.0% |
| 1.2 | Knowledge base has >=12 documents covering every required topic | Done | `kb/` | `transcripts/step2_pytest.txt` | | 12 docs, 4 sentences each |
| 1.3 | Two chunking strategies indexed into two separate ChromaDB collections, both retrieving sensibly | Done | `ola_agent/rag/chunking.py`, `ola_agent/rag/index.py` | `transcripts/step2_build_index.txt` | B9, B11 | F3 fixed: fixed-size chunk edges snap to spaces (38 / 48 chunks) |
| 1.4 | Grounded generation on >=5 in-scope queries plus 1 out-of-scope fallback | Done | `ola_agent/rag/generate.py` | `transcripts/step2_calibration.txt`, `transcripts/step2_grounded_demo_kb_fixed.txt` | B2-B7, B10, B13 | Thresholds 0.433 / 0.391 (after F3); F4 fixed (k < 1 rejected); F5 (competitor question) planned for step 3 guardrail |
| 1.5 | Precision/recall for both collections with per-query arithmetic and a numbers-cited recommendation | Done | `scripts/eval_retrieval.py` | `transcripts/step2_retrieval_eval.txt` | | kb_fixed 0.600/0.900 vs kb_sentence 0.567/0.900 (rerun after F3); kb_fixed still recommended |

## Part 2: CrewAI orchestration, tools, memory, guardrails

| # | Acceptance criterion | Status | Code file | Transcript | Attack test | Notes |
|---|---|---|---|---|---|---|
| 2.1 | `check_support_ticket_status` lookup plus designed, justified `escalation_score` | Not started | | | | |
| 2.2 | Crew of >=3 agents; RAG and lookup tools both invoked via `.kickoff()` | Not started | | | | |
| 2.3 | Multi-turn memory plus fresh-conversation transcript | Not started | | | | |
| 2.4 | Every crew response validates against a Pydantic schema | Not started | | | | |
| 2.5 | Input (PII/injection) and output (groundedness) guardrails each shown firing | Not started | | | | |

## Part 3: Evaluation, observability, FastAPI

| # | Acceptance criterion | Status | Code file | Transcript | Attack test | Notes |
|---|---|---|---|---|---|---|
| 3.1 | >=2 HTTP endpoints plus 1 WebSocket that survives a disconnect, with Pydantic models | Not started | | | | |
| 3.2 | One JSON-Lines log entry with trace ID per request | Not started | | | | |
| 3.3 | Accuracy, Grounding, Completeness, Safety per query for 15 queries, plus averages | Not started | | | | |

## Part 4: Resilience and governance

| # | Acceptance criterion | Status | Code file | Transcript | Attack test | Notes |
|---|---|---|---|---|---|---|
| 4.1 | Autogen review approves one draft unchanged and revises another, with Pydantic verdicts | Not started | | | | |
| 4.2 | Least autonomy shown; risk classification justified; budget cap rejects oversized request | Not started | | | | |
| 4.3 | Cache hit with before/after evidence | Not started | | | | |
