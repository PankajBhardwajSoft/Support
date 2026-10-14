"""The structured output every crew response must match (Part 2, Task 9; PRD D15)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TicketInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    found: bool
    status: str | None = None
    resolution_time_hours: float | None = None
    escalation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    escalate: bool | None = None
    message: str | None = None


class SupportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list, description="KB doc IDs the answer is taken from")
    ticket: TicketInfo | None = None
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Top policy-match similarity, clipped to [0, 1]")
