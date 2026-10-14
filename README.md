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
dataset.py        Part 1 Task 1: seeded support-ticket generator
kb/               12 knowledge-base documents
scripts/
  smoke_test.py   step 1: one-agent crew on the mock, network blocked
  download_model.py, build_index.py, calibrate_threshold.py, demo_grounded.py, eval_retrieval.py
tests/            unit tests and attack tests (all run with network blocked)
transcripts/      saved text output of every demonstration run
security/         attack-test reports, one per step
TRACKER.md        every requirement mapped to code and proof
```

## How the mock avoids the brief's two silent pitfalls

1. CrewAI's system prompt contains the example line `Observation: the result of the action`. The mock reads tool results **only from assistant-role messages** (its own turns, where CrewAI appends the real tool output), never from system or user messages. Test: `test_template_observation_is_not_a_result`.
2. Tool arguments are filled from each tool's **declared argument schema** (field names such as `record_id` or `query`), never from the tool's name. A tool named `rag_lookup` is never treated as the ticket lookup. Tests: `test_rag_lookup_name_does_not_make_it_a_ticket_tool`, `test_record_id_schema_is_filled_whatever_the_tool_is_called`.
