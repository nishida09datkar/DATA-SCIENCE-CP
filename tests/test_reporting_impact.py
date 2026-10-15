"""Financial impact comes from the detector's hourly scores, never from the LLM."""

from datetime import datetime

import pandas as pd
import pytest

from kpi_rca.config import ReportingConfig
from kpi_rca.reporting.impact import compute_financial_impact
from kpi_rca.schemas import AnomalyWindow, SegmentAttribute, SegmentFinding

CFG = ReportingConfig()  # SEV1 from 20,000 lost, SEV2 from 5,000


def make_scores():
    # Hourly buckets 12:00..17:00. The window covers 14:00, 15:00 and 16:00; 13:00 and 17:00
    # lose money too, so an off-by-one at either edge changes the totals.
    index = pd.date_range("2026-09-10 12:00", periods=6, freq="h", name="bucket")
    return pd.DataFrame(
        {
            "attempts": [1000, 1000, 1000, 1200, 1000, 1000],
            "successes": [800, 400, 500, 600, 820, 400],
            "revenue": [40000.0, 20000.0, 25000.0, 30000.0, 41000.0, 20000.0],
            "revenue_expected": [40000.0, 40000.0, 40000.0, 48000.0, 40000.0, 40000.0],
            "conversion_rate_expected": [0.8] * 6,
        },
        index=index,
    )


def make_window(start=datetime(2026, 9, 10, 14), end=datetime(2026, 9, 10, 17)):
    return AnomalyWindow(
        window_id="W-001",
        start=start,
        end=end,
        kpi_type="kpi_drop",
        n_buckets=3,
        if_score_min=-0.2,
        observed={"conversion_rate": 0.6, "revenue": 96000.0, "attempts": 3200.0},
        expected={"conversion_rate": 0.8, "revenue": 128000.0, "attempts": 3200.0},
    )


def make_finding():
    return SegmentFinding(
        rank=1,
        attributes=[SegmentAttribute(attribute="country", value="DE"), SegmentAttribute(attribute="os", value="iOS")],
        ep=0.9,
        purity=0.7,
        score=0.79,
        failure_rate_window=0.62,
        failure_rate_baseline=0.08,
        n_window=850,
        n_baseline=12000,
        excess_failures=460.0,
        p_value=1e-12,
        shap_contribution={"country=DE": 0.4, "os=iOS": 0.3},
        top_error_codes={"PAYPAL_SDK_TIMEOUT": 400},
    )


def one_bucket_shortfall(shortfall):
    index = pd.DatetimeIndex([datetime(2026, 9, 10, 14)], name="bucket")
    scores = pd.DataFrame(
        {
            "attempts": [1000],
            "successes": [800],
            "revenue": [50000.0 - shortfall],
            "revenue_expected": [50000.0],
            "conversion_rate_expected": [0.8],
        },
        index=index,
    )
    return scores, make_window(end=datetime(2026, 9, 10, 15))


def test_losses_are_summed_over_window_buckets_only_and_clipped_at_zero():
    impact = compute_financial_impact(make_window(), make_scores(), make_finding(), CFG)
    # 14:00 loses 15,000 and 300 orders, 15:00 loses 18,000 and 360 orders,
    # 16:00 beats its expectation, which must not cancel the earlier losses
    assert impact.lost_revenue == pytest.approx(33000.0)
    assert impact.lost_orders == pytest.approx(660.0)


def test_duration_rate_and_conversion_come_from_the_same_rows():
    impact = compute_financial_impact(make_window(), make_scores(), make_finding(), CFG)
    assert impact.currency == "USD"
    assert impact.duration_hours == pytest.approx(3.0)
    assert impact.revenue_at_risk_per_hour == pytest.approx(11000.0)
    assert impact.conversion_observed == pytest.approx(1920 / 3200)
    assert impact.conversion_expected == pytest.approx(2560 / 3200)


def test_segment_fields_come_from_the_top_finding():
    impact = compute_financial_impact(make_window(), make_scores(), make_finding(), CFG)
    assert impact.affected_attempts == 850
    assert impact.segment_failure_rate_window == pytest.approx(0.62)
    assert impact.segment_failure_rate_baseline == pytest.approx(0.08)


def test_without_a_finding_the_segment_fields_are_empty():
    impact = compute_financial_impact(make_window(), make_scores(), None, CFG)
    assert impact.affected_attempts is None
    assert impact.segment_failure_rate_window is None
    assert impact.segment_failure_rate_baseline is None


@pytest.mark.parametrize(
    "shortfall, severity",
    [(20000.0, "SEV1"), (19999.99, "SEV2"), (5000.0, "SEV2"), (4999.99, "SEV3"), (0.0, "SEV3")],
)
def test_severity_follows_the_lost_revenue_thresholds(shortfall, severity):
    scores, window = one_bucket_shortfall(shortfall)
    assert compute_financial_impact(window, scores, None, CFG).severity == severity


def test_severity_thresholds_come_from_config():
    scores, window = one_bucket_shortfall(60.0)
    cfg = ReportingConfig(sev1_lost_revenue=100.0, sev2_lost_revenue=50.0)
    assert compute_financial_impact(window, scores, None, cfg).severity == "SEV2"


def test_surge_above_expectation_costs_nothing():
    scores = make_scores()
    scores["revenue"] = scores["revenue_expected"] * 1.8
    scores["successes"] = scores["attempts"]
    impact = compute_financial_impact(make_window(), scores, None, CFG)
    assert impact.lost_revenue == 0.0 and impact.lost_orders == 0.0
    assert impact.severity == "SEV3"


def test_missing_score_columns_are_named():
    with pytest.raises(ValueError, match="revenue_expected"):
        compute_financial_impact(make_window(), make_scores().drop(columns="revenue_expected"), None, CFG)


def test_scores_must_be_indexed_by_bucket_time():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_financial_impact(make_window(), make_scores().reset_index(), None, CFG)


def test_window_outside_the_scores_is_rejected():
    window = make_window(start=datetime(2026, 9, 20, 14), end=datetime(2026, 9, 20, 17))
    with pytest.raises(ValueError, match="no score rows"):
        compute_financial_impact(window, make_scores(), None, CFG)
