"""Root-cause analysis on the small scenario: INC-T1 is {country=DE, os=iOS, payment_gateway=PayPal}."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from kpi_rca.rca import analyze_window, brute_force_window
from kpi_rca.rca.search import one_sided_p_value, slice_stats
from kpi_rca.schemas import AnomalyWindow, RCAResult

from conftest import SMALL_INCIDENT_START, SMALL_SURGE_START

INCIDENT = {("country", "DE"), ("os", "iOS"), ("payment_gateway", "PayPal")}


def make_window(start, hours: int, kpi_type: str = "kpi_drop", window_id: str = "W-001") -> AnomalyWindow:
    return AnomalyWindow(
        window_id=window_id,
        start=start,
        end=start + timedelta(hours=hours),
        kpi_type=kpi_type,
        n_buckets=hours,
        if_score_min=-0.1,
        observed={},
        expected={},
    )


def segment(finding) -> set[tuple[str, str]]:
    return {(a.attribute, a.value) for a in finding.attributes}


@pytest.fixture(scope="module")
def incident_window() -> AnomalyWindow:
    return make_window(SMALL_INCIDENT_START, 4)


@pytest.fixture(scope="module")
def shap_result(small_scenario, small_cfg, incident_window) -> RCAResult:
    return analyze_window(small_scenario.transactions, incident_window, small_cfg.rca)


@pytest.fixture(scope="module")
def brute_result(small_scenario, small_cfg, incident_window) -> RCAResult:
    return brute_force_window(small_scenario.transactions, incident_window, small_cfg.rca)


def test_shap_search_finds_the_incident_segment(shap_result, small_cfg):
    assert shap_result.method == "shap_search"
    assert shap_result.window_id == "W-001"
    top = shap_result.findings[0]
    assert top.rank == 1
    assert segment(top) == INCIDENT
    assert top.p_value < small_cfg.rca.alpha
    assert top.failure_rate_window > 0.8 > 0.2 > top.failure_rate_baseline
    assert next(iter(top.top_error_codes)) == "PAYPAL_SDK_TIMEOUT"
    assert len(top.top_error_codes) <= 3
    assert list(top.top_error_codes.values()) == sorted(top.top_error_codes.values(), reverse=True)


def test_finding_scores_follow_the_definitions(shap_result):
    for rank, f in enumerate(shap_result.findings, start=1):
        assert f.rank == rank
        assert f.score == pytest.approx(2 * f.ep * f.purity / (f.ep + f.purity))
        assert f.excess_failures > 0
        assert set(f.shap_contribution) == {f"{a.attribute}={a.value}" for a in f.attributes}
        assert all(v > 0 for v in f.shap_contribution.values())


def test_shap_outputs_use_attribute_value_keys(shap_result):
    fi = shap_result.feature_importance
    assert 0 < len(fi) <= 20
    assert list(fi.values()) == sorted(fi.values(), reverse=True)
    assert all("=" in k for k in fi)
    assert {"country=DE", "os=iOS", "payment_gateway=PayPal"} <= set(list(fi)[:5])
    pairs = {frozenset((p.feature_a, p.feature_b)) for p in shap_result.top_interactions[:3]}
    assert frozenset(("country=DE", "payment_gateway=PayPal")) in pairs


def test_results_round_trip_through_the_schema(shap_result, brute_result):
    for r in (shap_result, brute_result):
        assert RCAResult.model_validate(r.model_dump()) == r


def test_brute_force_agrees_but_scores_more_slices(shap_result, brute_result):
    assert brute_result.method == "brute_force"
    assert [segment(f) for f in brute_result.findings] == [segment(f) for f in shap_result.findings]
    assert brute_result.findings[0].ep == pytest.approx(shap_result.findings[0].ep)
    assert brute_result.slices_evaluated > 10 * shap_result.slices_evaluated
    assert brute_result.feature_importance == {} and brute_result.findings[0].shap_contribution == {}


def test_high_baseline_country_is_not_blamed(shap_result, brute_result, small_scenario, small_cfg, incident_window):
    """BR fails more than other countries every day; that is normal, not a cause."""
    tx = small_scenario.transactions
    assert (1 - tx.loc[tx["country"] == "BR", "success"].mean()) > (1 - tx["success"].mean())
    no_offset = analyze_window(tx, incident_window, small_cfg.rca, use_offset=False)
    for r in (shap_result, brute_result, no_offset):
        assert all(("country", "BR") not in segment(f) for f in r.findings)
    # Ablation A2: without the M0 offset, M1 spends part of its explanation on BR's normal rate.
    assert no_offset.feature_importance.get("country=BR", 0.0) > shap_result.feature_importance.get("country=BR", 0.0)


@pytest.mark.parametrize("method", [analyze_window, brute_force_window])
def test_quiet_window_and_benign_surge_have_no_findings(method, small_scenario, small_cfg):
    tx = small_scenario.transactions
    quiet = make_window(SMALL_INCIDENT_START - timedelta(days=1), 4, window_id="W-quiet")
    surge = make_window(SMALL_SURGE_START, 5, kpi_type="volume_surge", window_id="W-surge")
    for w in (quiet, surge):
        r = method(tx, w, small_cfg.rca)
        assert r.findings == [] and r.window_id == w.window_id


def test_window_without_baseline_returns_empty(small_scenario, small_cfg):
    first = small_scenario.transactions["timestamp"].min()
    r = analyze_window(small_scenario.transactions, make_window(first, 2), small_cfg.rca)
    assert r.findings == [] and r.slices_evaluated == 0


def test_other_kpi_drop_windows_leave_the_baseline(small_scenario, small_cfg, incident_window):
    from kpi_rca.rca.segment_model import prepare_window_data

    tx = small_scenario.transactions
    earlier = make_window(incident_window.start - timedelta(days=2), 3, window_id="W-000")
    surge = make_window(incident_window.start - timedelta(days=3), 3, kpi_type="volume_surge", window_id="W-S")
    plain = prepare_window_data(tx, incident_window, small_cfg.rca)
    excluded = prepare_window_data(tx, incident_window, small_cfg.rca, exclude_windows=[earlier, surge, incident_window])
    ts = tx["timestamp"]
    n_earlier = int(((ts >= earlier.start) & (ts < earlier.end)).sum())
    assert n_earlier > 0
    # Only the other kpi_drop is removed; the surge and the window itself are ignored.
    assert len(plain.y_baseline) - len(excluded.y_baseline) == n_earlier
    assert len(excluded.y_window) == len(plain.y_window)
    # The full entry point must accept it too (regression: read-only array under pandas Copy-on-Write).
    r = analyze_window(tx, incident_window, small_cfg.rca, exclude_windows=[earlier])
    assert segment(r.findings[0]) == INCIDENT


def test_bad_inputs_raise(small_scenario, small_cfg, incident_window):
    tx = small_scenario.transactions
    with pytest.raises(ValueError, match="missing columns"):
        analyze_window(tx.drop(columns=["os"]), incident_window, small_cfg.rca)
    empty = make_window(tx["timestamp"].max() + timedelta(days=30), 2)
    with pytest.raises(ValueError, match="no transactions"):
        brute_force_window(tx, empty, small_cfg.rca)


def test_slice_score_and_z_test_by_hand(small_cfg):
    # 100 rows, 60 failures, p0 = 0.1 each: excess = 50, expected successes = 90.
    s = slice_stats(
        ((0, 1),),
        n_w=100,
        fails_w=60,
        n_b=1000,
        fails_b=100,
        excess=50.0,
        expected_success=90.0,
        expected_fails=10.0,
        var_fails=9.0,
        total_excess=80.0,
        cfg=small_cfg.rca,
    )
    assert s.ep == pytest.approx(50 / 80)
    assert s.purity == pytest.approx(50 / 90)
    assert s.score == pytest.approx(2 * (50 / 80) * (50 / 90) / (50 / 80 + 50 / 90))
    pooled = 160 / 1100
    z = (0.6 - 0.1) / np.sqrt(pooled * (1 - pooled) * (1 / 100 + 1 / 1000))
    from scipy.stats import norm

    assert s.p_value == pytest.approx(norm.sf(z))
    # Too few baseline rows: model-based test, (60 - 10) / sqrt(9).
    assert one_sided_p_value(60, 100, 0, 5, 10.0, 9.0, min_rows=30) == pytest.approx(norm.sf(50 / 3))
    # No excess -> score 0.
    assert slice_stats(((0, 1),), n_w=10, fails_w=1, n_b=10, fails_b=1, excess=-0.5, expected_success=9,
                       expected_fails=1.5, var_fails=1.0, total_excess=80.0, cfg=small_cfg.rca).score == 0.0
