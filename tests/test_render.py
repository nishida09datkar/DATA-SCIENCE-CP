"""The incident report on disk: Markdown for people, JSON for machines, PNGs for the slides."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from kpi_rca.config import LLMConfig, ReportingConfig
from kpi_rca.reporting.context import build_incident_context
from kpi_rca.reporting.llm import TemplateNarrator, narrate_with_guardrails
from kpi_rca.reporting.models import FinancialImpact, IncidentReport, Provenance, RecommendedAction
from kpi_rca.reporting.render import render_markdown, save_report
from kpi_rca.schemas import AnomalyWindow, RCAResult, RetrievalResult, SegmentAttribute, SegmentFinding, TicketHit

CFG = ReportingConfig()
START, END = datetime(2026, 9, 10, 14), datetime(2026, 9, 10, 17)
QUOTE = "PayPal keeps timing out on my iPhone and the order never goes through."

WINDOW = AnomalyWindow(
    window_id="W-001",
    start=START,
    end=END,
    kpi_type="kpi_drop",
    n_buckets=3,
    if_score_min=-0.2,
    observed={"conversion_rate": 0.6, "revenue": 96000.0, "attempts": 3200.0},
    expected={"conversion_rate": 0.8, "revenue": 128000.0, "attempts": 3200.0},
)
FINDING = SegmentFinding(
    rank=1,
    attributes=[
        SegmentAttribute(attribute="country", value="DE"),
        SegmentAttribute(attribute="os", value="iOS"),
        SegmentAttribute(attribute="payment_gateway", value="PayPal"),
    ],
    ep=0.91,
    purity=0.7,
    score=0.79,
    failure_rate_window=0.62,
    failure_rate_baseline=0.08,
    n_window=850,
    n_baseline=12000,
    excess_failures=460.0,
    p_value=1e-12,
    shap_contribution={"payment_gateway=PayPal": 0.4, "os=iOS": 0.25, "country=DE": 0.1},
    top_error_codes={"PAYPAL_SDK_TIMEOUT": 400},
)
IMPACT = FinancialImpact(
    currency="USD",
    lost_revenue=33000.12,
    lost_orders=660.0,
    revenue_at_risk_per_hour=11000.04,
    duration_hours=3.0,
    conversion_observed=0.6,
    conversion_expected=0.8,
    affected_attempts=850,
    segment_failure_rate_window=0.62,
    segment_failure_rate_baseline=0.08,
    severity="SEV1",
)
RETRIEVAL = RetrievalResult(
    query="paypal sdk timeout",
    backend="lsa",
    search_start=datetime(2026, 9, 10, 12),
    search_end=datetime(2026, 9, 10, 23),
    hits=[
        TicketHit(
            rank=1,
            ticket_id="TK000001",
            created_at=datetime(2026, 9, 10, 14, 5),
            channel="chat",
            customer_country="DE",
            device_os="iOS",
            text=QUOTE,
            similarity=0.8,
            score=0.9,
        )
    ],
    n_candidates=40,
    similar_in_window=12,
    window_rate_per_hour=4.0,
    baseline_rate_per_hour=0.5,
    spike_ratio=8.0,
)


def make_report(findings=True, retrieval=True, llm=LLMConfig(provider="template")):
    rca = RCAResult(
        window_id="W-001",
        findings=[FINDING] if findings else [],
        feature_importance={},
        top_interactions=[],
        slices_evaluated=42,
        runtime_s=1.5,
    )
    hits = RETRIEVAL if retrieval else None
    context = build_incident_context(WINDOW, rca, hits, IMPACT, CFG)
    narrative, guardrails = narrate_with_guardrails(TemplateNarrator(), context, llm, CFG)
    return IncidentReport(
        incident_id="INC-20260910-1400",
        window=WINDOW,
        impact=IMPACT,
        findings=rca.findings,
        retrieval=hits,
        narrative=narrative,
        guardrails=guardrails,
        provenance=Provenance(
            narrator=guardrails.narrator_used,
            model=None,
            stage_timings_s={"detection": 1.25, "rca": 3.5},
            generated_at=datetime(2026, 9, 10, 18, 2),
        ),
    )


def make_scores():
    """Hourly scores around the window, as detection produces them (only the columns render reads)."""
    index = pd.date_range("2026-09-09 14:00", "2026-09-11 13:00", freq="h", name="bucket")
    inside = (index >= START) & (index < END)
    expected = 0.8 + 0.02 * np.sin(np.arange(len(index)) / 24 * 2 * np.pi)
    return pd.DataFrame(
        {
            "conversion_rate": np.where(inside, 0.6, expected),
            "conversion_rate_expected": expected,
            "attempts": 1000.0,
            "attempts_expected": 1000.0,
            "is_anomaly": inside,
        },
        index=index,
    )


# --- Markdown ------------------------------------------------------------------------------


def test_markdown_carries_the_memo_and_the_numbers_behind_it():
    report = make_report()
    md = render_markdown(report)
    assert md.startswith(f"# INC-20260910-1400: {report.narrative.headline}\n")
    assert report.narrative.executive_summary in md
    assert report.narrative.root_cause_explanation in md
    assert "USD 33,000" in md  # lost revenue in the impact table
    assert "country=DE & os=iOS & payment_gateway=PayPal" in md  # the finding's label
    assert f"> {QUOTE}" in md
    assert all(a.action in md for a in report.narrative.recommended_actions)


def test_markdown_records_how_the_narrative_was_produced():
    md = render_markdown(make_report())
    assert "Narrator: template" in md and "Guardrails: passed" in md
    assert "detection" in md and "1.25" in md  # stage timings


def test_markdown_shows_a_fallback_and_its_reason():
    report = make_report()
    report.guardrails.fallback_used = True
    report.guardrails.narrator_used = "template_fallback"
    report.guardrails.fallback_reason = "attempt 1: rate limit (429) still hit after the SDK's retries"
    md = render_markdown(report)
    assert "template_fallback" in md and "rate limit (429)" in md


def test_a_pipe_or_newline_in_a_cell_cannot_break_a_table():
    report = make_report()
    report.narrative.recommended_actions[0] = RecommendedAction(
        action="Fail over A | B\nthen re-test.", owner_team="SRE", priority="P0", rationale="x"
    )
    md = render_markdown(report)
    row = next(line for line in md.splitlines() if "Fail over" in line)
    assert "A \\| B then re-test." in row
    assert row.count("|") - row.count("\\|") == 5  # 4 columns -> 5 unescaped pipes


def test_markdown_without_findings_or_tickets():
    md = render_markdown(make_report(findings=False, retrieval=False))
    assert "No segment explained the anomaly." in md
    assert "No similar support tickets were found." in md


# --- files on disk ------------------------------------------------------------------------


def test_save_report_writes_markdown_json_and_both_plots(tmp_path):
    report = make_report()
    paths = save_report(report, make_scores(), tmp_path / "reports")
    assert set(paths) == {"markdown", "json", "kpi_plot", "shap_plot"}
    assert paths["markdown"].name == "INC-20260910-1400.md"
    assert paths["kpi_plot"].name == "INC-20260910-1400_kpi.png"
    for path in paths.values():
        assert path.parent == tmp_path / "reports" and path.stat().st_size > 0
    assert paths["kpi_plot"].read_bytes().startswith(b"\x89PNG")
    assert paths["markdown"].read_text(encoding="utf-8") == render_markdown(report)


def test_the_json_round_trips_to_the_same_report(tmp_path):
    report = make_report()
    paths = save_report(report, make_scores(), tmp_path, make_plots=False)
    text = paths["json"].read_text(encoding="utf-8")
    assert IncidentReport.model_validate_json(text) == report
    assert json.loads(text)["guardrails"]["narrator_used"] == "template"


def test_plots_can_be_switched_off(tmp_path):
    paths = save_report(make_report(), make_scores(), tmp_path, make_plots=False)
    assert set(paths) == {"markdown", "json"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["INC-20260910-1400.json", "INC-20260910-1400.md"]


def test_without_findings_there_is_no_shap_plot(tmp_path):
    paths = save_report(make_report(findings=False), make_scores(), tmp_path)
    assert set(paths) == {"markdown", "json", "kpi_plot"}


@pytest.mark.parametrize("column", ["conversion_rate", "conversion_rate_expected"])
def test_the_kpi_plot_needs_the_kpi_and_its_expected_column(tmp_path, column):
    with pytest.raises(ValueError, match=column):
        save_report(make_report(), make_scores().drop(columns=column), tmp_path)
