"""One question, end to end: input guardrails -> crew -> schema check -> groundedness check.

Matches PRD D14 for the parts built so far (budget cap, cache and the Autogen
review are added in later steps).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ola_agent import config
from ola_agent.crew import CrewRun, run_crew
from ola_agent.guardrails import GroundednessCheck, InputCheck, check_groundedness, check_input
from ola_agent.rag.generate import RECOMMENDED_COLLECTION, load_threshold
from ola_agent.schema import SupportResponse


@dataclass
class PipelineResult:
    question: str  # masked
    response: SupportResponse
    input_check: InputCheck
    crew: CrewRun | None = None
    output_check: GroundednessCheck | None = None
    guardrails_fired: list[str] = field(default_factory=list)


def ask(question: str, client=None) -> PipelineResult:
    check = check_input(question)
    fired = []
    if check.phones_masked:
        fired.append(f"input: masked {check.phones_masked} phone number(s)")
    if check.blocked:
        fired.append(f"input: blocked ({check.reason})")
        response = SupportResponse(answer=check.refusal, sources=[], ticket=None, grounded=False, confidence=0.0)
        return PipelineResult(check.text, response, check, guardrails_fired=fired)

    run = run_crew(check.text, client)
    output = check_groundedness(run.draft, run.search.get("chunks", []), load_threshold(RECOMMENDED_COLLECTION),
                                model_written=not config.MOCK_LLM)
    if output.fired:
        action = "refused" if output.refused else "trimmed"
        fired.append(f"output: groundedness {action} ({output.reason})")
    response = SupportResponse.model_validate(output.response.model_dump())
    return PipelineResult(check.text, response, check, run, output, fired)
