"""The incident context is the only thing the narrator sees: plain JSON, rounded, with ticket text fenced off."""

import json
from datetime import datetime

from kpi_rca.config import ReportingConfig
from kpi_rca.reporting.context import build_incident_context
from kpi_rca.reporting.guardrails import check_quote_grounding
from kpi_rca.reporting.models import FinancialImpact
from kpi_rca.schemas import (
    AnomalyWindow,
    RCAResult,
    RetrievalResult,
    SegmentAttribute,
    SegmentFinding,
    TicketHit,
)

CFG = ReportingConfig()
LONG_TEXT = "PayPal keeps timing out on my iPhone and the order never goes through at all."


def make_window(expected_attempts=3200.0):
    return AnomalyWindow(
        window_id="W-001",
        start=datetime(2026, 9, 10, 14),
        end=datetime(2026, 9, 10, 17),
        kpi_type="kpi_drop",
        n_buckets=3,
        if_score_min=-0.2,
        observed={"conversion_rate": 0.6, "revenue": 96000.0, "attempts": 3200.0},
        expected={"conversion_rate": 0.8, "revenue": 128000.0, "attempts": expected_attempts},
    )


def make_finding(failure_rate_baseline=0.08):
    return SegmentFinding(
        rank=1,
        attributes=[
            SegmentAttribute(attribute="country", value="DE"),
            SegmentAttribute(attribute="os", value="iOS"),
            SegmentAttribute(attribute="payment_gateway", value="PayPal"),
        ],
        ep=0.912345,
        purity=0.7,
        score=0.79,
        failure_rate_window=0.62,
        failure_rate_baseline=failure_rate_baseline,
        n_window=850,
        n_baseline=12000,
        excess_failures=460.04,
        p_value=1e-12,
        shap_contribution={"payment_gateway=PayPal": 0.4},
        top_error_codes={"PAYPAL_SDK_TIMEOUT": 400, "CARD_DECLINED": 12},
    )


def make_rca(findings=None):
    return RCAResult(
        window_id="W-001",
        findings=[make_finding()] if findings is None else findings,
        feature_importance={"payment_gateway=PayPal": 0.8},
        top_interactions=[],
        slices_evaluated=42,
        runtime_s=1.5,
    )


def make_hit(rank, text, **overrides):
    fields = dict(
        rank=rank,
        ticket_id=f"TK{rank:06d}",
        created_at=datetime(2026, 9, 10, 14, 5 * rank),
        channel="chat",
        customer_country="DE",
        device_os="iOS",
        text=text,
        similarity=0.81 - rank / 100,
        score=0.91 - rank / 100,
    )
    fields.update(overrides)
    return TicketHit(**fields)


def make_retrieval(hits=None, baseline_rate=0.5):
    window_rate = 12 / 11
    return RetrievalResult(
        query="paypal sdk timeout",
        backend="lsa",
        search_start=datetime(2026, 9, 10, 12),
        search_end=datetime(2026, 9, 10, 23),
        hits=[make_hit(1, LONG_TEXT), make_hit(2, "Checkout with PayPal fails every time.")] if hits is None else hits,
        n_candidates=40,
        similar_in_window=12,
        window_rate_per_hour=window_rate,
        baseline_rate_per_hour=baseline_rate,
        spike_ratio=window_rate / max(baseline_rate, 1e-9),
    )


IMPACT = FinancialImpact(
    currency="USD",
    lost_revenue=33000.123,
    lost_orders=660.04,
    revenue_at_risk_per_hour=11000.041,
    duration_hours=3.0,
    conversion_observed=0.6,
    conversion_expected=0.8,
    affected_attempts=850,
    segment_failure_rate_window=0.62,
    segment_failure_rate_baseline=0.08,
    severity="SEV1",
)


def build(window=None, rca=None, retrieval="default", cfg=CFG):
    retrieval = make_retrieval() if retrieval == "default" else retrieval
    return build_incident_context(window or make_window(), rca or make_rca(), retrieval, IMPACT, cfg)


def test_context_survives_a_json_round_trip_unchanged():
    ctx = build()
    assert json.loads(json.dumps(ctx)) == ctx


def test_incident_times_are_iso_strings_and_severity_comes_from_the_impact():
    incident = build()["incident"]
    assert incident["start"] == "2026-09-10T14:00:00"
    assert incident["end"] == "2026-09-10T17:00:00"
    assert incident["duration_hours"] == 3.0
    assert incident["severity"] == "SEV1"


def test_kpis_carry_observed_expected_and_changes():
    kpis = build()["kpis"]
    assert kpis["conversion_rate"] == {"observed": 0.6, "expected": 0.8, "abs_change": -0.2, "rel_change": -0.25}
    assert kpis["revenue"]["abs_change"] == -32000.0


def test_relative_change_is_empty_when_nothing_was_expected():
    assert build(window=make_window(expected_attempts=0.0))["kpis"]["attempts"]["rel_change"] is None


def test_money_is_rounded_to_cents():
    impact = build()["impact"]
    assert impact["currency"] == "USD"
    assert impact["lost_revenue"] == 33000.12
    assert impact["revenue_at_risk_per_hour"] == 11000.04


def test_top_finding_lists_its_attributes_for_the_segment_guardrail():
    finding = build()["findings"][0]
    assert finding["attributes"] == [
        {"attribute": "country", "value": "DE"},
        {"attribute": "os", "value": "iOS"},
        {"attribute": "payment_gateway", "value": "PayPal"},
    ]
    assert finding["failure_rate_ratio"] == 7.75
    assert finding["share_of_excess_failures"] == 0.9123
    assert finding["top_error_codes"] == {"PAYPAL_SDK_TIMEOUT": 400, "CARD_DECLINED": 12}


def test_failure_ratio_is_empty_without_a_baseline_rate():
    rca = make_rca(findings=[make_finding(failure_rate_baseline=0.0)])
    assert build(rca=rca)["findings"][0]["failure_rate_ratio"] is None


def test_no_findings_gives_an_empty_list():
    assert build(rca=make_rca(findings=[]))["findings"] == []


def test_ticket_excerpts_are_capped_in_number_and_cut_at_a_word():
    hits = [make_hit(1, LONG_TEXT), make_hit(2, "Second."), make_hit(3, "Third.")]
    ctx = build(retrieval=make_retrieval(hits=hits), cfg=ReportingConfig(excerpt_chars=40, max_ticket_excerpts=2))
    assert [t["ticket_id"] for t in ctx["tickets"]] == ["TK000001", "TK000002"]
    assert ctx["tickets"][0]["excerpt"] == "PayPal keeps timing out on my iPhone..."
    assert check_quote_grounding(["PayPal keeps timing out on my iPhone"], ctx) == []


def test_excerpt_collapses_whitespace():
    hits = [make_hit(1, "PayPal\n\n  failed   again.")]
    assert build(retrieval=make_retrieval(hits=hits))["tickets"][0]["excerpt"] == "PayPal failed again."


def test_customer_text_appears_only_under_the_untrusted_excerpt_key():
    ctx = build()
    for ticket in ctx["tickets"]:
        del ticket["excerpt"]
    assert "timing out" not in json.dumps(ctx)


def test_missing_ticket_metadata_stays_empty():
    hits = [make_hit(1, "PayPal failed.", customer_country=None, device_os=None)]
    ticket = build(retrieval=make_retrieval(hits=hits))["tickets"][0]
    assert ticket["customer_country"] is None and ticket["device_os"] is None


def test_ticket_spike_statistics_are_included():
    evidence = build()["ticket_evidence"]
    assert evidence["similar_in_window"] == 12
    assert evidence["window_rate_per_hour"] == 1.09
    assert evidence["spike_ratio"] == 2.18


def test_spike_ratio_is_empty_when_there_is_no_baseline():
    assert build(retrieval=make_retrieval(baseline_rate=0.0))["ticket_evidence"]["spike_ratio"] is None


def test_without_retrieval_there_is_no_ticket_evidence():
    ctx = build(retrieval=None)
    assert ctx["ticket_evidence"] is None and ctx["tickets"] == []
