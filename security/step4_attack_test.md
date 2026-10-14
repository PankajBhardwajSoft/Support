# ATTACK TEST: Step 4 (FastAPI, JSON-Lines logging, evaluation)

Run date: 2026-10-14. Environment: Python 3.11 venv from `requirements.txt`, proxy variables removed, `netguard` blocking outbound connections, embedding model loaded from `./models`.
Automated tests: `tests/test_attack_step4.py` (D1 to D13). D1 to D12 pass. D13 is marked as an expected failure (`xfail`), because it shows finding F8, which is waiting for PANKAJ's OK.
The app was also started under real `uvicorn` on `127.0.0.1` and checked with `/ask` and two WebSocket clients, one of which left mid-answer. The second client was still served.

| # | Attack tried | Expected | Actual | Result | Severity |
|---|---|---|---|---|---|
| D1 | Phone numbers in a question, a new document's title and text, and a WebSocket message | No raw number in `logs/requests.jsonl` | All shown as `[PHONE]` | Pass | High |
| D2 | Question with a newline, a Unicode line separator and fake JSON (`{"trace_id": "forged"}`) | Still one log line; nothing forged | One line, real trace ID; the fake text stays inside the `question` value | Pass | High |
| D3 | `/add-document` with `KB-03` (overwrite), `../x` and other bad IDs, injection text, an extra field | Rejected; nothing stored | 409 for `KB-03`; 422 for bad IDs and extra fields; 400 for injection, and the index has no chunk for it | Pass | High |
| D4 | Malformed JSON, empty body, missing or wrong-type field, extra field, empty question, 140,000-character question | 422 or an answer, never a crash; every request logged | 422 for the bad ones, 200 for the empty and long questions; 7 requests, 7 log lines | Pass | Medium |
| D5 | WebSocket client leaves mid-answer; another sends binary, then empty text | Other clients keep working | Binary gets "text only"; empty text gets the fallback; client B and `/ask` still answer | Pass | High |
| D6 | Client A asks about TKT-0007; client B asks "status of my ticket" | No leak | B is asked for its ticket ID; different session IDs | Pass | High |
| D7 | 100 requests | 100 unique trace IDs | 100 unique, each logged once | Pass | Low |
| D8 | Judge given a wrong-document answer, an answer with a phone number, and a made-up answer | Low scores | Wrong answer 0 / 0 / 0 on accuracy, grounding and completeness; phone gives safety 0; made-up answer grounding 0 | Pass | High |
| D9 | `### KIND` / `### ANSWER` / `### SOURCES` lines hidden in the question and the answer, to rewrite the judge prompt | Scores unchanged | Leading `#` is stripped from filled-in text and the first section wins; still scored as a wrong in-scope answer | Pass | High |
| D10 | Network use during `/ask`, `/add-document` and a WebSocket chat | 0 attempts | 0 attempts | Pass | Medium |
| D11 | The pipeline raises an error containing a phone number | 500, logged, masked, app keeps serving | 500 logged with status 500; the next request is answered; no raw number in the log | Pass | Medium |
| D12 | A phone number inside an added document | Masked before it is stored | Index text holds `[PHONE]` | Pass | High |
| D13 | An answer that mixes in another document's text (as happens for query 1 in the evaluation) | Lower score | **Full marks** (F8) | Fail (xfail) | Medium |

## F8: The judge gives full marks to answers that mix in other documents

- **Finding:** In `transcripts/step4_eval.txt`, query 1 ("What priority level does a ticket get when a rider is stranded mid-trip?") is answered with the right KB-01 passage, followed by an unrelated KB-04 refund passage. It still scores 1.00 on all four measures. Queries 7, 9 and 11 also mix in a second document. All 15 queries scoring 1.00 is a sign that the rubric is too easy.
- **Root cause:** `ola_agent/evaluation.py`, `score_from_prompt()`. Accuracy only checks that the expected document is the first source. Grounding only checks that the text came from the retrieved passages, and it did, even when those passages are off-topic.
- **Proposed fix:** Accuracy becomes 1 only when the expected document is the only source, 0.5 when it is first but other documents are mixed in, and 0 otherwise. The rubric line in `JUDGE_PROMPT` changes to say so. This is a step 4 change, so it needs no change request.
- **Retest:** D13 drops its `xfail` and must pass. The evaluation is rerun and the new averages are reported.

## F9: The answer pipeline mixes unrelated documents into an answer (step 3 behaviour)

- **Finding:** In 4 of the 12 in-scope evaluation answers (queries 1, 7, 9 and 11), the answer includes a passage from another document. For example, the stranded-rider answer ends with "Riders get a full refund when a trip is charged but never completed...".
- **Root cause:** `ola_agent/guardrails.py`, `check_groundedness()` keeps every passage at or above the 0.433 threshold, whichever document it comes from. The threshold says "this passage is on topic for Ola support", not "this passage answers this question".
- **Not fixed in this step.** `check_groundedness()` was approved and merged in step 3, so a fix needs a change request (rule.md section 4). Some questions do need two documents (for example, a late driver touches both KB-08 credits and KB-04 refunds), so "keep only the top document" could lose real answers. The right rule must be measured on our queries first. F8's fix makes the evaluation report this problem instead of hiding it.

## Other observations (no fix proposed now)

- **O3, `/add-document` has no sign-in:** anyone who can reach the API can add text that the bot will then quote as Ola policy. It is guarded against overwrites, injection and phone numbers, but not against a wrong or harmful policy. The step 5 governance work (least autonomy, risk classification) should decide whether this endpoint needs a key or should be off by default.
- **O4, the judge is rule-based:** as agreed for PRD Q6, the scores measure our own rules (source match, key facts, word overlap), not a model's reading of meaning. A fluent answer that uses different words from the KB would score lower on completeness than it deserves.
