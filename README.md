# Ola Domain Support Agent (CrewAI)

**Track completed: Ola (Business Operations / Customer Support).**

A support agent that answers Ola support-policy questions from a knowledge base and checks support-ticket status from a generated dataset. Built with CrewAI, an Autogen review stage, LangChain session memory, ChromaDB and SentenceTransformers, behind a FastAPI backend.

> Status: step 2 of 6 (Part 1: dataset and RAG core). See `TRACKER.md` for progress against every requirement.

## Runs offline with zero API keys

- Every language-model call uses a deterministic mock (`MOCK_LLM`, on by default). No API key and no network are needed.
- The mock extends CrewAI's own extension point, `crewai.llms.base_llm.BaseLLM` (`ola_agent/mock_llm.py`).
- **CrewAI telemetry is disabled.** `ola_agent/config.py` sets `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` before CrewAI is imported, and overrides any other value already set in the shell.
- Setting `MOCK_LLM=false` stops the program with an error; a real LLM is not wired in.

## Dataset design choices (Part 1, Task 1)

Reproduce with `python dataset.py` (output: `transcripts/step2_dataset.txt`).

| Choice | Value |
|---|---|
| Seed | `42` (`random.Random(42)`); first seed tried, passed every check |
| Records | 50, `TKT-0001` to `TKT-0050` |
| Category weights | Billing 0.25, Technical Issue 0.25, Account Access 0.20, Product Defect 0.15, General Inquiry 0.15 |
| Status weights | Open 0.20, In Progress 0.25, Escalated 0.10, Resolved 0.25, Closed 0.20 |
| `resolution_time_hours` | Uniform 1.0 to 168.0, 1 decimal. Simple account fixes close within an hour, while defect and billing investigations can take up to a week. |
| `days_since_created` | Integer 0 to 30 |
| `escalated` | True with probability 0.15; a ticket with status `Escalated` is always `escalated=True` (generator rule, never a hand edit) |
| Amount range | Not applicable: no ticket field in the brief holds an amount |

Result: Technical Issue 15, Billing 11, Product Defect 11, General Inquiry 7, Account Access 6; Resolved 12, In Progress 12, Open 11, Closed 11, Escalated 4; escalated 13/50 = **26.0%** (band 10 to 30%).

## Knowledge base and retrieval (Part 1, Tasks 2 to 5)

- `kb/`: 12 documents, one per required topic, 4 sentences each, written for this project with made-up rules.
- Two chunking strategies (`ola_agent/rag/chunking.py`): fixed-size windows of up to 200 characters with 50 overlap, edges moved to the nearest space so no word is cut (38 chunks) and one chunk per sentence (48 chunks).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, run locally on CPU, normalized.
- Each strategy has its own ChromaDB collection, `kb_fixed` and `kb_sentence`, cosine distance, written with `collection.upsert()`.

### "I don't know" threshold, measured (Task 4)

Top-1 cosine similarity from `scripts/calibrate_threshold.py` (`transcripts/step2_calibration.txt`):

| Query | Type | kb_fixed | kb_sentence |
|---|---|---|---|
| How quickly will I get a reply to a P1 ticket? | in-scope | 0.674 | 0.748 |
| I was charged twice for one ride, can I get a refund? | in-scope | 0.681 | 0.719 |
| When does a ticket get escalated to Tier 2? | in-scope | 0.711 | 0.756 |
| How long do you keep my support chat history? | in-scope | 0.676 | 0.751 |
| What compensation do I get if my driver is very late? | in-scope | 0.489 | 0.472 |
| What is the capital of France? | out-of-scope | 0.117 | 0.135 |
| Can you recommend a good biryani recipe? | out-of-scope | 0.134 | 0.118 |
| How do I reset my Netflix password? | out-of-scope | 0.378 | 0.310 |
| **Threshold** = midpoint of lowest in-scope and highest out-of-scope | | **0.433** | **0.391** |

Both thresholds sit well below the common 0.5 to 0.7 tutorial presets: a 0.5 preset would have wrongly refused the in-scope late-driver question. Thresholds are saved in `rag_thresholds.json`. The demo (`transcripts/step2_grounded_demo_kb_fixed.txt`) answers all 5 in-scope queries from retrieved text only and returns the fallback for "What is the capital of France?" (0.117).

### Chunking comparison and recommendation (Task 5)

Document-level precision and recall over the same 5 queries, top-3 chunks mapped to parent documents and deduplicated (`transcripts/step2_retrieval_eval.txt` has the per-query arithmetic):

| Collection | Mean precision | Mean recall |
|---|---|---|
| `kb_fixed` | (0.500 + 0.500 + 1.000 + 0.500 + 0.500) / 5 = **0.600** | (1 + 1 + 1 + 1 + 0.5) / 5 = **0.900** |
| `kb_sentence` | (0.500 + 0.500 + 1.000 + 0.500 + 0.333) / 5 = **0.567** | (1 + 1 + 1 + 1 + 0.5) / 5 = **0.900** |

**Recommendation: deploy `kb_fixed`.** Both strategies reach the same mean recall (0.900), but `kb_fixed` has higher mean precision (0.600 vs 0.567). The difference comes from the late-driver question: its 200-character chunks pull in 1 wrong document there, against 2 for sentence chunks (precision 0.500 vs 0.333); the other 4 queries score the same. The margin is small, so the 15-query evaluation in step 4 will recheck it. These numbers were measured after fix F3 (whole-word chunk edges, `security/step2_attack_test.md`); before it, `kb_fixed` scored 0.700.

### One-time model download

The embedding model must be downloaded once with internet access:

```powershell
python scripts\download_model.py
```

It is saved to `models/all-MiniLM-L6-v2` (git-ignored). Every other script loads it from there with `HF_HUB_OFFLINE=1` and the network blocked.

### Run Part 1

```powershell
python dataset.py
python scripts\build_index.py
python scripts\calibrate_threshold.py
python scripts\demo_grounded.py
python scripts\eval_retrieval.py
```

Known limitation: the embedding model is English-only, so non-English questions get the "I don't know" fallback.

## Crew, tools, memory and guardrails (Part 2, Tasks 6 to 10)

### Ticket lookup and escalation score (Task 6)

`check_support_ticket_status(record_id)` (`ola_agent/tools/ticket_lookup.py`) returns `status`, `resolution_time_hours` and `escalation_score`. Unknown or malformed IDs return `found: false` instead of an error.

```
escalation_score = 0.5 * escalated + 0.5 * (days_since_created / 30)
```

- Both parts are in [0, 1], so the score is too. Older tickets score higher, because a ticket left open longer is more likely to need escalating.
- **Why equal weights:** with 0.6 / 0.4, every escalated ticket on our data scored at least 0.613 and every other ticket at most 0.400. Any threshold then just copies the `escalated` flag, which is a bare boolean. With 0.5 / 0.5, a 30-day-old ticket that was never escalated (0.500) reaches a fresh escalated one (0.517).
- **Recommended threshold: 0.50**, the 75th percentile of the score over our 50 tickets (nearest rank). It flags 14 tickets: all 13 escalated ones, plus TKT-0016, which is 30 days old and was never escalated. Distribution and list: `transcripts/step3_escalation_threshold.txt`.
- **Limitation:** the brief defines the score from `escalated` and age only, so a Closed ticket can still score high.

### The crew (Task 7)

`ola_agent/crew.py` runs three agents in order with `.kickoff()`, all on `MockLLM`:

| Agent | Tool | Job |
|---|---|---|
| Retrieval Agent | `search_ola_policies(query)` over `kb_fixed` | Finds policy passages |
| Lookup Agent | `check_support_ticket_status(record_id)`, held by **no other agent** | Looks up the ticket, if the question has an ID |
| Response Composer | none | Combines both results into one draft |

The mock fills tool arguments from each tool's schema. A question without a `TKT-` ID gives the Lookup Agent nothing to fill, so it does not call the tool. `transcripts/step3_crew.txt` shows the RAG tool used on a policy question and the lookup tool on a ticket question.

### Session memory (Task 8)

`ola_agent/memory.py` uses LangChain's `InMemoryChatMessageHistory` with `RunnableWithMessageHistory`, one history per `session_id`, in memory only. A follow-up such as "How long will it take to resolve?" uses the last ticket ID in that session. A new session has none, so the agent asks for the ID. See `transcripts/step3_memory.txt`. The `LangChainDeprecationWarning` printed there is expected, as the brief says.

### Structured output (Task 9)

Every crew response is a `SupportResponse` (`ola_agent/schema.py`): `answer`, `sources`, `ticket`, `grounded`, `confidence`. It is set as the Composer task's `output_pydantic` and validated again in code with `model_validate`. Extra fields, an empty answer or a confidence outside [0, 1] raise an error.

### Guardrails (Task 10)

| Side | Guardrail | Rule |
|---|---|---|
| Input | Phone masking | Indian mobile numbers (optional `+91`, `91` or `0`; spaces or dashes allowed) become `[PHONE]` before the crew or memory sees the text |
| Input | Prompt injection | Pattern rules on Unicode-normalized text (e.g. "ignore previous instructions", "reveal your system prompt") |
| Input | Competitor scope | A question naming another ride company (Uber, Rapido, BluSmart, Meru, inDrive, Bolt, Namma Yatri, Lyft) gets "I don't know", because Ola's policies don't apply to them. A fixed list can't catch every company name without a real model |
| Output | Groundedness | The answer is rebuilt only from passages at or above the 0.433 threshold that come from the best-matching document (in document order, without repeated overlap), plus the ticket result. With neither, the answer is "I don't know". Limit: a question that truly needs two documents gets only the best one (F9) |

Each guardrail fires on a deliberate case in `transcripts/step3_guardrails.txt`.

### Run Part 2

```powershell
python scripts\escalation_threshold.py
python scripts\demo_crew.py
python scripts\demo_memory.py
python scripts\demo_guardrails.py
```

## Web API, logging and evaluation (Part 3, Tasks 11 to 13)

### FastAPI app (Task 11)

`ola_agent/api.py`. Start it with `uvicorn ola_agent.api:app` and open `http://127.0.0.1:8000/docs` to try it.

| Endpoint | Request | Response | What it does |
|---|---|---|---|
| `POST /ask` | `AskRequest(question)` | `AskResponse(trace_id, answer, sources, ticket, grounded, confidence, guardrails_fired)` | One question through guardrails, crew and groundedness check |
| `POST /add-document` | `AddDocumentRequest(doc_id, title, text)` | `AddDocumentResponse(trace_id, doc_id, collection, chunks_added)` | Chunks a new document and adds it to `kb_fixed` |
| `WebSocket /ws/chat` | text messages | JSON per turn: `trace_id`, `session_id`, `answer`, `sources`, `grounded` | Multi-turn chat with session memory, one session per connection |

- `/add-document` accepts only new IDs like `KB-13` (no paths, no overwriting the 12 originals: 409). Phone numbers in the text are masked before storing, and text with an injection pattern is rejected (400).
- The WebSocket catches `WebSocketDisconnect`, including a client that leaves mid-answer. Only that connection ends; other clients keep being served.
- `transcripts/step4_api.txt` shows both endpoints, the new document being found, and client A leaving while client B carries on with its memory.

### JSON-Lines logging (Task 12)

`ola_agent/request_log.py` writes one line per HTTP request and per WebSocket message to `logs/requests.jsonl` (not committed). Fields: `ts`, `trace_id`, `endpoint`, `session_id`, `question`, `answer`, `guardrails_fired`, `sources`, `grounded`, `status`, `duration_ms`. The trace ID is also returned in the response body and the `X-Trace-ID` header. Every text field goes through the same phone masking as the model input right before it is written, so no raw phone number reaches disk. Requests rejected before the handler (for example a 422) are logged too.

### Evaluation of 15 queries (Task 13)

`ola_agent/evaluation.py`, run with `scripts/run_eval.py`. The test set has one question per KB topic (12) plus 3 out-of-scope or edge cases (an off-topic question, a competitor question and a prompt injection). Each in-scope question was labelled with its expected document and 2 key facts from the KB text before any run.

The judge is a written prompt (`JUDGE_PROMPT`) filled in with the question, labels, retrieved passages and answer. Under `MOCK_LLM` it is read by `JudgeLLM`, a `MockLLM` that applies the rules in that prompt and returns JSON checked against the `JudgeScores` model. **The scores measure these fixed rules, not a real model's opinion** (agreed for PRD Q6).

| Score (0 to 1) | Rule |
|---|---|
| Accuracy | In scope: 1 if the expected document is the only source, 0.5 if it is first but other documents are mixed in, else 0. Out of scope: 1 if refused |
| Grounding | Share of answer sentences with at least 80 percent of their words in the retrieved passages; a refusal scores 1 |
| Completeness | In scope: share of the expected key facts in the answer. Out of scope: 1 if refused |
| Safety | 0 if the answer has a phone number or answers an out-of-scope question |

Averages on our 15 queries (`transcripts/step4_eval.txt`): accuracy 1.00, grounding 1.00, completeness 1.00, safety 1.00. Before fix F9, 4 of the 12 answers also included a passage from an unrelated document and average accuracy was 0.87 (see `security/step4_attack_test.md`). Because the answers are copied from the KB and the labels come from the same text, these scores show the right document was found and nothing off-topic was added; they do not measure how well a free-text answer is worded. A negative control (the data-retention answer given to the Tier 2 question) scores 0 on accuracy, grounding and completeness, so the judge does mark wrong answers down.

### Run Part 3

```powershell
python scripts\demo_api.py
python scripts\run_eval.py
uvicorn ola_agent.api:app
```

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
  rag/            Part 1: KB loading, chunking, embeddings, ChromaDB, grounded answers
  tools/          Part 2: policy search tool and ticket lookup tool
  crew.py         Part 2: Retrieval, Lookup and Composer agents
  schema.py       Part 2: SupportResponse, the structured output
  guardrails.py   Part 2: phone masking, injection and scope checks, groundedness
  pipeline.py     Part 2: one question end to end (guardrails, crew, checks)
  memory.py       Part 2: LangChain session memory
  api.py          Part 3: FastAPI app, two HTTP endpoints and a WebSocket
  request_log.py  Part 3: JSON-Lines request log with trace IDs, phones masked
  evaluation.py   Part 3: 15-query test set, judge prompt and mock judge
dataset.py        Part 1 Task 1: seeded support-ticket generator
kb/               12 knowledge-base documents
scripts/
  smoke_test.py   step 1: one-agent crew on the mock, network blocked
  download_model.py, build_index.py, calibrate_threshold.py, demo_grounded.py, eval_retrieval.py
  escalation_threshold.py, demo_crew.py, demo_memory.py, demo_guardrails.py
  demo_api.py, run_eval.py
tests/            unit tests and attack tests (all run with network blocked)
transcripts/      saved text output of every demonstration run
security/         attack-test reports, one per step
TRACKER.md        every requirement mapped to code and proof
```

## How the mock avoids the brief's two silent pitfalls

1. CrewAI's system prompt contains the example line `Observation: the result of the action`. The mock reads tool results **only from assistant-role messages** (its own turns, where CrewAI appends the real tool output), never from system or user messages. Test: `test_template_observation_is_not_a_result`.
2. Tool arguments are filled from each tool's **declared argument schema** (field names such as `record_id` or `query`), never from the tool's name. A tool named `rag_lookup` is never treated as the ticket lookup. Tests: `test_rag_lookup_name_does_not_make_it_a_ticket_tool`, `test_record_id_schema_is_filled_whatever_the_tool_is_called`.
