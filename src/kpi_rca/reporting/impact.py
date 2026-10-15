"""Deterministic financial impact of an anomaly window.

Every number is computed from the detector's hourly scores; the LLM only ever
sees the result. Losses are counted per bucket and clipped at zero, so an hour
that beats its expectation does not cancel out an hour that lost money.
"""

from __future__ import annotations

import pandas as pd

from kpi_rca.config import ReportingConfig
from kpi_rca.reporting.models import FinancialImpact
from kpi_rca.schemas import AnomalyWindow, SegmentFinding

REQUIRED_COLUMNS = ("attempts", "successes", "revenue", "revenue_expected", "conversion_rate_expected")


def compute_financial_impact(
    window: AnomalyWindow, scores: pd.DataFrame, finding: SegmentFinding | None, cfg: ReportingConfig
) -> FinancialImpact:
    rows = _window_rows(window, scores)
    expected_successes = rows["attempts"] * rows["conversion_rate_expected"]
    lost_revenue = float((rows["revenue_expected"] - rows["revenue"]).clip(lower=0).sum())
    lost_orders = float((expected_successes - rows["successes"]).clip(lower=0).sum())
    attempts = float(rows["attempts"].sum())
    duration_hours = (window.end - window.start).total_seconds() / 3600
    return FinancialImpact(
        currency=cfg.currency,
        lost_revenue=lost_revenue,
        lost_orders=lost_orders,
        revenue_at_risk_per_hour=lost_revenue / duration_hours,
        duration_hours=duration_hours,
        # a window with no traffic at all has no conversion to report
        conversion_observed=float(rows["successes"].sum()) / attempts if attempts else 0.0,
        conversion_expected=float(expected_successes.sum()) / attempts if attempts else 0.0,
        affected_attempts=finding.n_window if finding else None,
        segment_failure_rate_window=finding.failure_rate_window if finding else None,
        segment_failure_rate_baseline=finding.failure_rate_baseline if finding else None,
        severity=_severity(lost_revenue, cfg),
    )


def _window_rows(window: AnomalyWindow, scores: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise ValueError("scores must be indexed by a DatetimeIndex of bucket start times")
    missing = [c for c in REQUIRED_COLUMNS if c not in scores.columns]
    if missing:
        raise ValueError(f"scores is missing columns {missing}")
    rows = scores[(scores.index >= window.start) & (scores.index < window.end)]
    if rows.empty:
        raise ValueError(f"no score rows inside window {window.window_id} ({window.start} to {window.end})")
    return rows


def _severity(lost_revenue: float, cfg: ReportingConfig) -> str:
    if lost_revenue >= cfg.sev1_lost_revenue:
        return "SEV1"
    if lost_revenue >= cfg.sev2_lost_revenue:
        return "SEV2"
    return "SEV3"
