"""One question, end to end (PRD D14):

input guardrails -> budget cap -> cache check (crew on a miss) -> schema check
-> groundedness check -> Autogen review -> response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ola_agent import config
from ola_agent.cache import CACHE, cache_key
from ola_agent.crew import CrewRun, run_crew
from ola_agent.governance import BUDGET_REFUSAL, check_budget
from ola_agent.guardrails import GroundednessCheck, InputCheck, check_groundedness, check_input, ticket_sentence
from ola_agent.rag.generate import FALLBACK_ANSWER, RECOMMENDED_COLLECTION, load_threshold
from ola_agent.review import Verdict, review
from ola_agent.schema import SupportResponse


@dataclass
class PipelineResult:
    question: str  # masked
    response: SupportResponse
    input_check: InputCheck
    crew: CrewRun | None = None
    output_check: GroundednessCheck | None = None
    guardrails_fired: list[str] = field(default_factory=list)
    over_budget: bool = False
    cache_hit: bool = False
    verdict: Verdict | None = None


def ask(question: str, client=None) -> PipelineResult:
    check = check_input(question)
    fired = []
    if check.phones_masked:
        fired.append(f"input: masked {check.phones_masked} phone number(s)")
    if check.blocked:
        fired.append(f"input: blocked ({check.reason})")
        response = SupportResponse(answer=check.refusal, sources=[], ticket=None, grounded=False, confidence=0.0)
        return PipelineResult(check.text, response, check, guardrails_fired=fired)

    budget = check_budget(check.text)
    if not budget.ok:
        fired.append(f"runtime: budget cap ({budget.reason})")
        response = SupportResponse(answer=BUDGET_REFUSAL, sources=[], ticket=None, grounded=False, confidence=0.0)
        return PipelineResult(check.text, response, check, guardrails_fired=fired, over_budget=True)

    key = cache_key(check.text, client, config.MOCK_LLM)
    run = CACHE.get(key)
    hit = run is not None
    if hit:
        fired.append("cache: hit (crew not run)")
    else:
        run = run_crew(check.text, client)
        CACHE.put(key, run)

    threshold = load_threshold(RECOMMENDED_COLLECTION)
    chunks = run.search.get("chunks", [])
    output = check_groundedness(run.draft, chunks, threshold, model_written=not config.MOCK_LLM)
    if output.fired:
        action = "refused" if output.refused else "trimmed"
        fired.append(f"output: groundedness {action} ({output.reason})")
    response = SupportResponse.model_validate(output.response.model_dump())

    verdict = None
    if not output.refused:
        # The review sees the same evidence the groundedness check kept.
        evidence = [c["text"] for c in chunks if c["doc_id"] in response.sources and c["similarity"] >= threshold]
        if response.ticket is not None:
            evidence.append(ticket_sentence(response.ticket))
        verdict = review(response.answer, evidence)
        if not verdict.approved:
            fired.append(f"review: revised ({verdict.reason})")
            response = _apply_revision(response, verdict)
    return PipelineResult(check.text, response, check, run, output, fired, cache_hit=hit, verdict=verdict)


def _apply_revision(response: SupportResponse, verdict: Verdict) -> SupportResponse:
    if verdict.final_answer == FALLBACK_ANSWER:
        return SupportResponse(answer=FALLBACK_ANSWER, sources=[], ticket=None, grounded=False,
                               confidence=response.confidence)
    return response.model_copy(update={"answer": verdict.final_answer})
