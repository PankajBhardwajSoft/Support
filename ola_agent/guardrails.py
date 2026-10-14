"""Input and output guardrails (Part 2, Task 10).

Input side (before the crew sees anything):
- phone-number masking (PRD D11: Indian mobile, optional +91 / 0 prefix),
- prompt-injection detection (PRD D12: pattern rules, works with no model),
- competitor scope (step 2 finding F5: Ola's policies are not another company's).

Output side (after the crew): groundedness. The answer is rebuilt only from
evidence that passes the check: policy passages at or above the calibrated
"I don't know" threshold, and the ticket tool's result. If nothing passes,
the answer is "I don't know".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ola_agent.rag.generate import FALLBACK_ANSWER
from ola_agent.schema import SupportResponse, TicketInfo
from ola_agent.tools.ticket_lookup import ESCALATION_THRESHOLD

PHONE_MASK = "[PHONE]"
# Optional +91 / 91 / 0 prefix, then 10 digits starting 6-9; a space or dash may sit between digits.
PHONE_PATTERN = re.compile(r"(?<![\d+])(?:\+?91[\s-]?|0)?[6-9](?:[\s-]?\d){9}(?!\d)")

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}\b(instructions?|rules|prompts?|guidelines)\b",
    r"\b(reveal|print|show|repeat|leak|tell me)\b.{0,30}\bsystem\s*prompt\b",
    r"\bsystem\s*prompt\b.{0,30}\b(is|says|contains)\b",
    r"\byou are now\b",
    r"\b(developer|dan|jailbreak)\s*mode\b",
    r"\bjailbreak\b",
    r"\bpretend (to be|you are)\b",
]
_INJECTION = [re.compile(p) for p in INJECTION_PATTERNS]

# Other ride-hailing brands (F5, F6). A fixed list cannot be complete without a real model; see README.
COMPETITORS = ("uber", "rapido", "blusmart", "blu smart", "meru", "indrive", "in drive", "bolt", "namma yatri", "lyft")
_COMPETITOR = re.compile(r"\b(" + "|".join(re.escape(c).replace(r"\ ", r"\s?") for c in COMPETITORS) + r")\b")

INJECTION_REFUSAL = "I can't help with that request. I can answer questions about Ola support policies or a ticket ID."


def _normalize(text: str) -> str:
    """Lower-case, Unicode-normalized, single-spaced text for pattern checks."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")  # drop zero-width characters
    return re.sub(r"\s+", " ", text)


def mask_phones(text: str) -> tuple[str, int]:
    return PHONE_PATTERN.subn(PHONE_MASK, text)


def find_injection(text: str) -> str | None:
    norm = _normalize(text)
    for pattern in _INJECTION:
        match = pattern.search(norm)
        if match:
            return match.group(0)
    return None


def find_competitor(text: str) -> str | None:
    match = _COMPETITOR.search(_normalize(text))
    return match.group(0) if match else None


@dataclass
class InputCheck:
    text: str  # the masked question; the only version anything downstream sees
    phones_masked: int = 0
    blocked: bool = False
    reason: str | None = None
    refusal: str | None = None


def check_input(question: str) -> InputCheck:
    masked, count = mask_phones(question or "")
    check = InputCheck(text=masked, phones_masked=count)
    if not masked.strip():
        check.blocked, check.reason, check.refusal = True, "empty question", FALLBACK_ANSWER
    elif (hit := find_injection(masked)) is not None:
        check.blocked, check.reason, check.refusal = True, f"prompt injection: '{hit}'", INJECTION_REFUSAL
    elif (hit := find_competitor(masked)) is not None:
        check.blocked, check.reason, check.refusal = True, f"out of scope: names another company ('{hit}')", FALLBACK_ANSWER
    return check


_MIN_OVERLAP = 10


def join_passages(chunks: list[dict]) -> str:
    """Join passages in document order, dropping text that overlapping chunks repeat (F7).

    Documents keep the order of their best match; inside a document, chunks go
    in chunk-number order. Only retrieved text is used.
    """
    doc_rank = {}
    for c in sorted(chunks, key=lambda c: -c["similarity"]):
        doc_rank.setdefault(c["doc_id"], len(doc_rank))
    ordered = sorted({c["chunk_id"]: c for c in chunks}.values(),
                     key=lambda c: (doc_rank[c["doc_id"]], c["chunk_id"]))
    pieces = []
    for prev, cur in zip([None] + ordered, ordered):
        text = cur["text"]
        if prev is not None and prev["doc_id"] == cur["doc_id"]:
            overlap = next((k for k in range(min(len(prev["text"]), len(text)), _MIN_OVERLAP - 1, -1)
                            if prev["text"].endswith(text[:k])), 0)
            text = text[overlap:].lstrip()
        if text:
            pieces.append(text)
    return " ".join(pieces)


def ticket_sentence(ticket: TicketInfo) -> str:
    if not ticket.found:
        return ticket.message or f"I couldn't find ticket {ticket.record_id}."
    side = "at or above" if ticket.escalate else "below"
    return (
        f"Ticket {ticket.record_id} is {ticket.status}, with a resolution time of "
        f"{ticket.resolution_time_hours} hours. Its escalation score is {ticket.escalation_score:.3f}, "
        f"{side} the escalation threshold of {ESCALATION_THRESHOLD:.2f}."
    )


@dataclass
class GroundednessCheck:
    response: SupportResponse
    fired: bool
    refused: bool = False
    reason: str | None = None
    dropped_chunks: list[str] = field(default_factory=list)


def check_groundedness(draft: SupportResponse, chunks: list[dict], threshold: float) -> GroundednessCheck:
    """Keep only supported evidence; refuse with "I don't know" when none is left.

    Supported means at or above the threshold AND from the best-matching
    document (F9): passages from other documents are on topic for Ola support
    but, on our labelled questions, never part of the answer.
    """
    above = [c for c in chunks if c["similarity"] >= threshold]
    best_doc = max(above, key=lambda c: c["similarity"])["doc_id"] if above else None
    supported = [c for c in above if c["doc_id"] == best_doc]
    below = [c["chunk_id"] for c in chunks if c["similarity"] < threshold]
    other_doc = [c["chunk_id"] for c in above if c["doc_id"] != best_doc]
    dropped = below + other_doc
    parts = [ticket_sentence(draft.ticket)] if draft.ticket else []
    if supported:
        parts.append(join_passages(supported))
    if not parts:
        top = max((c["similarity"] for c in chunks), default=0.0)
        refused = SupportResponse(answer=FALLBACK_ANSWER, sources=[], ticket=None, grounded=False, confidence=draft.confidence)
        return GroundednessCheck(refused, True, True, f"top policy match {top:.3f} is below the {threshold} threshold and no ticket was looked up", dropped)
    response = SupportResponse(
        answer=" ".join(parts),
        sources=list(dict.fromkeys(c["doc_id"] for c in supported)),
        ticket=draft.ticket,
        grounded=True,
        confidence=draft.confidence,
    )
    notes = []
    if below:
        notes.append(f"dropped {len(below)} passage(s) below the {threshold} threshold")
    if other_doc:
        notes.append(f"dropped {len(other_doc)} passage(s) from other documents than {best_doc}")
    reason = "; ".join(notes) or None
    return GroundednessCheck(response, bool(dropped), False, reason, dropped)
