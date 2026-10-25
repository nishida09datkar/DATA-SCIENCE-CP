"""Shared result schemas (INTERFACE SPEC v1, section 1).

All timestamps are naive UTC datetimes. Field names, types and defaults here are
part of the cross-team contract; change them only through the spec owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel

KPI_COLUMNS = ("conversion_rate", "revenue", "attempts")


class SegmentAttribute(BaseModel):
    attribute: str  # country | os | device_type | payment_gateway | app_version
    value: str


class AnomalyWindow(BaseModel):
    window_id: str  # "W-001", ... in time order
    start: datetime  # first flagged bucket start
    end: datetime  # EXCLUSIVE: last flagged bucket start + 1 bucket
    kpi_type: Literal["kpi_drop", "volume_surge"]
    n_buckets: int
    if_score_min: float  # min IsolationForest decision_function in the window (lower = more anomalous)
    observed: dict[str, float]  # keys = KPI_COLUMNS; conversion = sum(successes)/sum(attempts); others = sums
    expected: dict[str, float]  # same keys, from STL trend+seasonal (expected conversion weighted by attempts)
    detected_at: datetime | None = None  # set by the streaming replay; None in batch mode
    # Additive summary fields (v1.1): defaults keep old constructors valid.
    duration_hours: float = 0.0  # end - start in hours (n_buckets in bucket-size units)
    peak_anomaly_score: float = 0.0  # max anomaly_score in the window (higher = more anomalous)
    # Signed expected-vs-observed gap of the primary KPI: (expected - observed)/expected of
    # conversion_rate for kpi_drop (positive = the KPI fell), of attempts for volume_surge
    # (positive = traffic rose). None when the expectation is 0 or not finite.
    drop_pct: float | None = None
    reason: str = ""  # human-readable reason of the window's peak bucket (see detection.detector.point_reason)


class SegmentFinding(BaseModel):
    rank: int  # 1 = top cause
    attributes: list[SegmentAttribute]
    ep: float  # explanatory power: share of the window's excess failures inside the segment
    purity: float  # excess failures / expected successes inside the segment
    score: float  # harmonic mean(ep, purity)
    failure_rate_window: float  # within the segment
    failure_rate_baseline: float  # within the segment
    n_window: int  # segment attempts in the window
    n_baseline: int  # segment attempts in the baseline period
    excess_failures: float  # sum over the segment's window rows of (y - p0)
    p_value: float  # two-proportion z-test, window vs baseline
    shap_contribution: dict[str, float]  # "attribute=value" -> mean SHAP (log-odds) over the segment's window rows
    top_error_codes: dict[str, int]  # error_code -> count among the segment's window failures; <= 3, descending

    def label(self) -> str:
        return " & ".join(f"{a.attribute}={a.value}" for a in self.attributes)


class InteractionPair(BaseModel):
    feature_a: str
    feature_b: str
    strength: float


class RCAResult(BaseModel):
    window_id: str
    findings: list[SegmentFinding]  # ranked; may be empty
    feature_importance: dict[str, float]  # "attribute=value" -> mean |SHAP|, top 20
    top_interactions: list[InteractionPair]
    slices_evaluated: int
    runtime_s: float
    method: Literal["shap_search", "brute_force"] = "shap_search"


class TicketHit(BaseModel):
    rank: int  # 1-based, after re-ranking
    ticket_id: str
    created_at: datetime
    channel: str
    customer_country: str | None = None
    device_os: str | None = None
    text: str  # full text; context.py truncates it
    similarity: float  # raw cosine (inner product of L2-normalised vectors)
    score: float  # similarity + metadata boosts; the sort key


class RetrievalResult(BaseModel):
    query: str
    backend: str  # "sentence_transformers" | "lsa"
    search_start: datetime  # window.start - pad_before_h
    search_end: datetime  # window.end + pad_after_h
    hits: list[TicketHit]
    n_candidates: int  # tickets in [search_start, search_end)
    similar_in_window: int  # of those, how many have similarity >= cfg.similarity_threshold
    window_rate_per_hour: float  # similar tickets/hour inside the search range
    baseline_rate_per_hour: float  # similar tickets/hour outside the search range
    spike_ratio: float  # window_rate / max(baseline_rate, 1e-9)


@dataclass
class DetectionResult:
    # scores: index is a DatetimeIndex named "bucket" (hourly, sorted, gap-free). Columns:
    #   attempts, successes, failures, conversion_rate, revenue, avg_latency_ms (observed)
    #   {k}_expected and {k}_resid_z for k in KPI_COLUMNS
    #   if_score (float, IsolationForest decision_function, lower = more anomalous)
    #   anomaly_score (float, = -if_score, HIGHER = more anomalous; the streaming replay fills it too)
    #   is_anomaly (bool); label ("normal" | "kpi_drop" | "volume_surge")
    #   reason ("": empty for normal buckets; a human-readable detection reason for flagged ones)
    #   imputed (bool, carried over from the raw KPI frame: buckets with interpolated rates)
    scores: pd.DataFrame
    windows: list[AnomalyWindow]
