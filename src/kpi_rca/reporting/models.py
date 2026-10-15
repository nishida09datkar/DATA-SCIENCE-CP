"""Pydantic models for the incident memo.

MemoNarrative is the ONLY thing an LLM is allowed to produce. It is kept to
closed objects (str / Literal / lists of nested models, no free-form dicts) so
it works as a structured-output schema. The limits below are written into the
field descriptions (what the LLM is told) and enforced by
reporting.guardrails (what is checked), so the two cannot drift apart.

Everything else here (FinancialImpact, ...) is computed deterministically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from kpi_rca.schemas import AnomalyWindow, RetrievalResult, SegmentFinding

HEADLINE_MAX_CHARS = 120
ACTION_MAX_WORDS = 20
MAX_EVIDENCE = 3
MAX_ACTIONS = 4


class FinancialImpact(BaseModel):
    currency: str
    lost_revenue: float  # sum over window buckets of max(0, revenue_expected - revenue)
    lost_orders: float  # sum over window buckets of max(0, attempts * conversion_rate_expected - successes)
    revenue_at_risk_per_hour: float  # lost_revenue / duration_hours
    duration_hours: float
    conversion_observed: float  # sum(successes) / sum(attempts) in the window
    conversion_expected: float  # sum(attempts * conversion_rate_expected) / sum(attempts)
    affected_attempts: int | None  # top finding's n_window; None without a finding
    segment_failure_rate_window: float | None
    segment_failure_rate_baseline: float | None
    severity: Literal["SEV1", "SEV2", "SEV3"]


class RecommendedAction(BaseModel):
    action: str = Field(description=f"One imperative sentence, at most {ACTION_MAX_WORDS} words.")
    owner_team: str = Field(description="Team that should own the action, e.g. 'Payments', 'Mobile', 'SRE'.")
    priority: Literal["P0", "P1", "P2"] = Field(description="P0 = do now, P1 = today, P2 = this week.")
    rationale: str = Field(description="One sentence tying the action to the evidence in the context.")


class MemoNarrative(BaseModel):
    headline: str = Field(
        description=f"One line, at most {HEADLINE_MAX_CHARS} characters, naming the broken segment and the impact."
    )
    executive_summary: str = Field(
        description="Two or three sentences: what broke, since when, and what it costs. "
        "Every number must be copied from the context JSON."
    )
    root_cause_explanation: str = Field(
        description="Name every attribute value of the top finding and cite its failure rate vs baseline."
    )
    customer_evidence: list[str] = Field(
        description=f"0-{MAX_EVIDENCE} verbatim quotes copied exactly from the ticket excerpts in the context."
    )
    recommended_actions: list[RecommendedAction] = Field(description=f"1-{MAX_ACTIONS} actions, most urgent first.")
    confidence: Literal["low", "medium", "high"]


class GuardrailReport(BaseModel):
    """How the final narrative was obtained and whether it passed reporting.guardrails."""

    passed: bool  # the final narrative passed every check
    attempts: int  # narrator calls made (the template fallback is not counted)
    violations: list[str]  # violations of the final narrative
    attempt_violations: list[list[str]]  # violations of each checked draft, in order
    narrator_used: str  # "claude" | "template" | "template_fallback"
    fallback_used: bool
    fallback_reason: str | None = None


class Provenance(BaseModel):
    narrator: str  # GuardrailReport.narrator_used
    model: str | None = None  # the configured LLM; None for the template
    stage_timings_s: dict[str, float]  # pipeline stage -> seconds
    generated_at: datetime  # naive UTC


class IncidentReport(BaseModel):
    """Everything one incident's report needs; render.py writes it as Markdown and JSON."""

    incident_id: str  # "INC-YYYYMMDD-HHMM" from the window start
    window: AnomalyWindow
    impact: FinancialImpact
    findings: list[SegmentFinding]
    retrieval: RetrievalResult | None = None
    narrative: MemoNarrative
    guardrails: GuardrailReport
    provenance: Provenance
