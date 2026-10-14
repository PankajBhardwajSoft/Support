"""Ticket lookup tool with a designed escalation score (Part 2, Task 6).

escalation_score = 0.5 * escalated + 0.5 * (days_since_created / 30)

Both parts are in [0, 1], so the score is too. Equal weights let age matter:
a 30-day-old ticket that was never escalated (0.500) can reach a fresh
escalated ticket (0.517 on our data). With 0.6 / 0.4 every escalated ticket
outscored every other one, so the age part never changed a decision.

ESCALATION_THRESHOLD is the 75th percentile of the score over our 50 tickets
(scripts/escalation_threshold.py prints the distribution; see README).
"""

from __future__ import annotations

import json
import re
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from dataset import SUPPORT_TICKETS

ESCALATED_WEIGHT = 0.5
RECENCY_WEIGHT = 0.5
MAX_DAYS = 30  # days_since_created ranges 0 to 30 in dataset.py
ESCALATION_THRESHOLD = 0.50
THRESHOLD_PERCENTILE = 75

_RECORD_ID = re.compile(r"^TKT-\d{4}$")
_TICKETS_BY_ID = {t["record_id"]: t for t in SUPPORT_TICKETS}


def escalation_score(escalated: bool, days_since_created: int) -> float:
    recency = min(max(days_since_created, 0), MAX_DAYS) / MAX_DAYS
    return round(ESCALATED_WEIGHT * float(escalated) + RECENCY_WEIGHT * recency, 3)


def check_support_ticket_status(record_id: str) -> dict[str, Any]:
    """Look up one ticket. Bad or unknown IDs return found=False, never raise."""
    if not isinstance(record_id, str) or not _RECORD_ID.match(record_id.strip()):
        return {"record_id": str(record_id)[:40], "found": False, "message": "Ticket IDs look like TKT-0001."}
    record_id = record_id.strip()
    ticket = _TICKETS_BY_ID.get(record_id)
    if ticket is None:
        return {"record_id": record_id, "found": False, "message": f"No ticket {record_id} was found."}
    score = escalation_score(ticket["escalated"], ticket["days_since_created"])
    return {
        "record_id": record_id,
        "found": True,
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "escalation_score": score,
        "escalate": score >= ESCALATION_THRESHOLD,
    }


class TicketLookupInput(BaseModel):
    record_id: str = Field(description="Ticket ID in the form TKT-0001")


class TicketLookupTool(BaseTool):
    name: str = "check_support_ticket_status"
    description: str = "Look up a support ticket's status, resolution time and escalation score by its ticket ID."
    args_schema: type[BaseModel] = TicketLookupInput
    _calls: list[str] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[str]:
        return self._calls

    def _run(self, record_id: str) -> str:
        self._calls.append(record_id)
        return json.dumps(check_support_ticket_status(record_id))
