"""The incident context: the ONLY input the narrator (LLM or template) sees.

It is plain JSON built from the deterministic results, with numbers rounded to
what a memo would quote. The guardrails treat every value in it as a fact,
except ticket text: customer-written words appear only under the "excerpt"
key, which reporting.guardrails treats as untrusted (quotable, never a fact).
"""

from __future__ import annotations

from kpi_rca.config import ReportingConfig
from kpi_rca.reporting.models import FinancialImpact
from kpi_rca.schemas import KPI_COLUMNS, AnomalyWindow, RCAResult, RetrievalResult, SegmentFinding, TicketHit

KPI_DIGITS = {"conversion_rate": 4, "revenue": 2, "attempts": 0}


def build_incident_context(
    window: AnomalyWindow,
    rca: RCAResult,
    retrieval: RetrievalResult | None,
    impact: FinancialImpact,
    cfg: ReportingConfig,
) -> dict:
    return {
        "incident": {
            "window_id": window.window_id,
            "kpi_type": window.kpi_type,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),  # exclusive
            "timezone": "UTC",
            "duration_hours": impact.duration_hours,
            "severity": impact.severity,
        },
        "kpis": {k: _kpi_change(window, k) for k in KPI_COLUMNS if k in window.observed and k in window.expected},
        "impact": {
            "currency": impact.currency,
            "lost_revenue": round(impact.lost_revenue, 2),
            "lost_orders": round(impact.lost_orders, 1),
            "revenue_at_risk_per_hour": round(impact.revenue_at_risk_per_hour, 2),
        },
        "findings": [_finding(f) for f in rca.findings],
        "ticket_evidence": _ticket_evidence(retrieval) if retrieval else None,
        "tickets": [_ticket(h, cfg.excerpt_chars) for h in retrieval.hits[: cfg.max_ticket_excerpts]]
        if retrieval
        else [],
    }


def _kpi_change(window: AnomalyWindow, kpi: str) -> dict:
    observed, expected = window.observed[kpi], window.expected[kpi]
    digits = KPI_DIGITS[kpi]
    return {
        "observed": round(observed, digits),
        "expected": round(expected, digits),
        "abs_change": round(observed - expected, digits),
        "rel_change": round(observed / expected - 1, 4) if expected else None,
    }


def _finding(f: SegmentFinding) -> dict:
    baseline = f.failure_rate_baseline
    return {
        "rank": f.rank,
        "segment": f.label(),
        "attributes": [{"attribute": a.attribute, "value": a.value} for a in f.attributes],
        "failure_rate_window": round(f.failure_rate_window, 4),
        "failure_rate_baseline": round(baseline, 4),
        "failure_rate_ratio": round(f.failure_rate_window / baseline, 2) if baseline > 0 else None,
        "share_of_excess_failures": round(f.ep, 4),  # the finding's explanatory power
        "attempts_in_window": f.n_window,
        "excess_failures": round(f.excess_failures, 1),
        "top_error_codes": dict(f.top_error_codes),
    }


def _ticket_evidence(r: RetrievalResult) -> dict:
    return {
        "search_start": r.search_start.isoformat(),
        "search_end": r.search_end.isoformat(),
        "n_candidates": r.n_candidates,
        "similar_in_window": r.similar_in_window,
        "window_rate_per_hour": round(r.window_rate_per_hour, 2),
        "baseline_rate_per_hour": round(r.baseline_rate_per_hour, 2),
        # with no similar tickets outside the range the ratio divides by ~0: not a number worth quoting
        "spike_ratio": round(r.spike_ratio, 2) if r.baseline_rate_per_hour > 0 else None,
    }


def _ticket(hit: TicketHit, max_chars: int) -> dict:
    return {
        "ticket_id": hit.ticket_id,
        "created_at": hit.created_at.isoformat(),
        "channel": hit.channel,
        "customer_country": hit.customer_country,
        "device_os": hit.device_os,
        "similarity": round(hit.similarity, 2),
        "excerpt": _excerpt(hit.text, max_chars),
    }


def _excerpt(text: str, max_chars: int) -> str:
    """Whitespace-collapsed text, at most max_chars long including the '...' that marks a cut."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[: max(max_chars - 3, 0)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]  # never end on half a word
    return cut.rstrip() + "..."
