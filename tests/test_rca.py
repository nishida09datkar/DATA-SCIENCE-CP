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

import pandas as pd


def test_m0_baseline_calibration(small_cfg):
    """Test A: M0 baseline probability p0 is calibrated to empirical baseline rates."""
    from kpi_rca.rca.segment_model import (
        baseline_failure_probability,
        fit_baseline_model,
        prepare_window_data,
    )

    rng = np.random.default_rng(42)
    start = SMALL_INCIDENT_START - timedelta(days=5)
    records = []
    # Segment A (low risk ~ 10%), Segment B (high risk ~ 40%)
    for i in range(1200):
        t = start + timedelta(hours=i * 0.1)
        is_b = i % 2 == 1
        country = "BR" if is_b else "US"
        fail_p = 0.40 if is_b else 0.10
        success = int(rng.random() >= fail_p)
        records.append({
            "transaction_id": f"tx-{i}",
            "timestamp": t,
            "country": country,
            "os": "iOS",
            "device_type": "mobile",
            "payment_gateway": "Card",
            "app_version": "1.0",
            "success": success,
            "error_code": "NONE" if success else "ERR",
            "amount": 10.0,
            "latency_ms": 100.0,
        })
    df = pd.DataFrame(records)
    window = make_window(start + timedelta(days=4), 4)
    data = prepare_window_data(df, window, small_cfg.rca)
    m0 = fit_baseline_model(data, small_cfg.rca)
    p0 = baseline_failure_probability(data, m0, small_cfg.rca, use_offset=True)

    us_mask = data.window_codes[:, 0] == data.categories["country"].index("US")
    br_mask = data.window_codes[:, 0] == data.categories["country"].index("BR")

    assert us_mask.sum() > 0 and br_mask.sum() > 0
    p0_us = p0[us_mask].mean()
    p0_br = p0[br_mask].mean()

    # Empirical rates are 0.10 and 0.40; M0 estimates should be well calibrated
    assert 0.05 < p0_us < 0.20
    assert 0.30 < p0_br < 0.50
    assert p0_br > p0_us + 0.15


def test_m1_offset_semantics(small_cfg):
    """Test B: M1 init_score = logit(p0) prevents attributing baseline heterogeneity to window change."""
    from kpi_rca.rca.segment_model import (
        baseline_failure_probability,
        change_margin,
        fit_baseline_model,
        fit_change_model,
        prepare_window_data,
    )

    rng = np.random.default_rng(123)
    start = SMALL_INCIDENT_START - timedelta(days=5)
    records = []
    # Baseline: country A = 10% fail, country B = 45% fail
    # Window: country A = 10% fail, country B = 45% fail (identical rates -> no incident!)
    for i in range(1200):
        t = start + timedelta(hours=i * 0.1)
        is_b = i % 2 == 1
        country = "BR" if is_b else "US"
        fail_p = 0.45 if is_b else 0.10
        success = int(rng.random() >= fail_p)
        records.append({
            "transaction_id": f"tx-{i}",
            "timestamp": t,
            "country": country,
            "os": "Android",
            "device_type": "mobile",
            "payment_gateway": "Card",
            "app_version": "1.0",
            "success": success,
            "error_code": "NONE" if success else "TIMEOUT",
            "amount": 25.0,
            "latency_ms": 150.0,
        })
    df = pd.DataFrame(records)
    window = make_window(start + timedelta(days=4), 4)
    data = prepare_window_data(df, window, small_cfg.rca)

    m0 = fit_baseline_model(data, small_cfg.rca)
    p0 = baseline_failure_probability(data, m0, small_cfg.rca, use_offset=True)

    # Fit M1 with offset
    m1 = fit_change_model(data, p0, small_cfg.rca, use_offset=True)
    margin = change_margin(m1, data)

    # In the window, neither segment changed compared to baseline.
    # Therefore, M1 change margin f(x) for the high-baseline segment (BR) should be around 0,
    # NOT strongly positive.
    br_mask = data.window_codes[:, 0] == data.categories["country"].index("BR")
    assert margin[br_mask].mean() < 0.2

    # And full analyze_window returns no findings (no false incident on BR)
    result = analyze_window(df, window, small_cfg.rca, use_offset=True)
    assert len(result.findings) == 0


def test_correlated_nested_candidate_recovery(small_cfg):
    """Test C: Correlated/nested attributes whose SHAP credit splits are recovered via greedy extension."""
    from kpi_rca.rca import brute_force_window

    rng = np.random.default_rng(999)
    start = SMALL_INCIDENT_START - timedelta(days=4)
    records = []
    # Baseline: everything ~5% failure rate
    # Window: only (country=DE, os=iOS) spikes to 85% failure rate
    w_start = start + timedelta(days=3)
    w_end = w_start + timedelta(hours=4)
    for i in range(8000):
        t = start + timedelta(hours=i * 0.01)
        country = rng.choice(["US", "DE", "GB"])
        os = rng.choice(["iOS", "Android"])
        gateway = rng.choice(["PayPal", "Stripe"])
        in_window = (t >= w_start and t < w_end)
        is_target = (country == "DE" and os == "iOS")
        if in_window and is_target:
            fail_p = 0.85
        else:
            fail_p = 0.05
        success = int(rng.random() >= fail_p)
        records.append({
            "transaction_id": f"tx-{i}",
            "timestamp": t,
            "country": country,
            "os": os,
            "device_type": "mobile",
            "payment_gateway": gateway,
            "app_version": "1.0",
            "success": success,
            "error_code": "NONE" if success else "ERR_FAIL",
            "amount": 50.0,
            "latency_ms": 200.0,
        })
    df = pd.DataFrame(records)
    window = make_window(w_start, 4)

    shap_res = analyze_window(df, window, small_cfg.rca)
    brute_res = brute_force_window(df, window, small_cfg.rca)

    assert len(shap_res.findings) >= 1
    top_finding = shap_res.findings[0]
    top_attrs = {(a.attribute, a.value) for a in top_finding.attributes}
    assert ("country", "DE") in top_attrs and ("os", "iOS") in top_attrs
    # Evaluation count must remain bounded and much lower than brute force (< 15%)
    assert shap_res.slices_evaluated < 0.15 * brute_res.slices_evaluated


def test_negative_global_excess_with_localized_spike(small_cfg):
    """Test D: RCA recovers a localized spike even when overall window excess is negative (macro drift)."""
    rng = np.random.default_rng(777)
    start = SMALL_INCIDENT_START - timedelta(days=5)
    records = []
    # 75% background (country=US), 25% target (country=DE)
    # Baseline:
    #   US: failure rate 40%
    #   DE: failure rate 5%
    # Window (macro shift where US drops to 10%, but DE spikes to 85%):
    #   US excess: negative
    #   DE excess: positive
    #   Total window excess <= 0
    w_start = start + timedelta(days=4)
    w_end = w_start + timedelta(hours=6)

    for i in range(15000):
        t = start + timedelta(hours=i * 0.008)
        is_window = (t >= w_start and t < w_end)
        is_target = (rng.random() < 0.25)
        country = "DE" if is_target else "US"
        os = "Android"
        gateway = "Card"

        if not is_window:
            fail_p = 0.05 if is_target else 0.40
        else:
            fail_p = 0.85 if is_target else 0.10

        success = int(rng.random() >= fail_p)
        records.append({
            "transaction_id": f"tx-{i}",
            "timestamp": t,
            "country": country,
            "os": os,
            "device_type": "mobile",
            "payment_gateway": gateway,
            "app_version": "1.0",
            "success": success,
            "error_code": "NONE" if success else "ERR_FAIL",
            "amount": 20.0,
            "latency_ms": 100.0,
        })
    df = pd.DataFrame(records)
    window = make_window(w_start, 6)

    res = analyze_window(df, window, small_cfg.rca)

    # Must NOT return zero findings merely because global excess <= 0
    assert len(res.findings) >= 1
    top = res.findings[0]
    top_attrs = {(a.attribute, a.value) for a in top.attributes}
    assert ("country", "DE") in top_attrs
    assert top.excess_failures > 0
    assert top.ep > 0
    assert top.p_value < small_cfg.rca.alpha
    # Benign segment US must NOT be a finding
    assert not any(("country", "US") in {(a.attribute, a.value) for a in f.attributes} for f in res.findings)


def test_rees46_adapter_end_to_end_rca():
    """Test E: REES46 in-memory fixture adapts sessions and runs through RCA with valid attributes."""
    from kpi_rca.config import RCAConfig
    from kpi_rca.data.adapters import inject_incident, rees46_transactions

    rng = np.random.default_rng(42)
    # Naive UTC datetime to match rees46_transactions format
    start = pd.Timestamp("2019-10-01 00:00:00")
    raw_events = []
    # Create 2000 sessions over 8 days
    # REES46 valid attributes: category, brand, price_band
    categories = ["electronics.smartphone", "appliances.kitchen.refrigerator", "computers.notebook"]
    brands = ["samsung", "apple", "xiaomi"]
    prices = [150.0, 500.0, 1200.0]

    for s_id in range(2000):
        session_uuid = f"sess-{s_id:04d}"
        t = start + pd.Timedelta(hours=s_id * 0.09)
        cat = categories[s_id % 3]
        brand = brands[(s_id // 3) % 3]
        price = prices[(s_id // 9) % 3]
        user_id = 1000 + s_id

        # Baseline: ~75% purchase success rate
        has_purchase = (s_id % 4 != 0)

        # Raw event 1: cart
        raw_events.append({
            "event_time": t,
            "event_type": "cart",
            "product_id": 100000 + s_id,
            "category_id": 200000 + (s_id % 3),
            "category_code": cat,
            "brand": brand,
            "price": price,
            "user_id": user_id,
            "user_session": session_uuid,
        })
        if has_purchase:
            raw_events.append({
                "event_time": t + pd.Timedelta(minutes=5),
                "event_type": "purchase",
                "product_id": 100000 + s_id,
                "category_id": 200000 + (s_id % 3),
                "category_code": cat,
                "brand": brand,
                "price": price,
                "user_id": user_id,
                "user_session": session_uuid,
            })

    raw_df = pd.DataFrame(raw_events)
    # Adapt to transaction DataFrame
    tx = rees46_transactions(raw_df)

    # 1. Exactly one transaction per user_session
    assert len(tx) == 2000
    assert tx["transaction_id"].nunique() == 2000

    # 2. Check schema and REES46 valid attributes
    expected_cols = {"transaction_id", "timestamp", "amount", "success", "error_code", "category", "brand", "price_band"}
    assert expected_cols.issubset(set(tx.columns))

    # 3. Verify success/failure mapping
    # Failed sessions have NO_PURCHASE error code
    failed_tx = tx[tx["success"] == 0]
    assert (failed_tx["error_code"] == "NO_PURCHASE").all()

    # 4. Inject an incident on segment brand=samsung in the window
    w_start = start + pd.Timedelta(days=7)
    w_end = w_start + pd.Timedelta(hours=12)
    tx_injected, hit = inject_incident(
        tx,
        segment={"brand": "samsung"},
        start=w_start.to_pydatetime(),
        end=w_end.to_pydatetime(),
        failure_rate=0.95,
        rng=rng,
    )
    assert hit.sum() > 0

    # 5. Run RCA with REES46 configuration attributes
    cfg = RCAConfig(
        attributes=["category", "brand", "price_band"],
        baseline_days=7.0,
        min_segment_rows=5,
        min_ep=0.05,
    )
    window = AnomalyWindow(
        window_id="W-REES46",
        start=w_start,
        end=w_end,
        kpi_type="kpi_drop",
        n_buckets=12,
        if_score_min=-0.1,
        observed={},
        expected={},
    )
    rca_res = analyze_window(tx_injected, window, cfg)
    assert rca_res.method == "shap_search"
    assert len(rca_res.findings) >= 1
    top_attrs = {(a.attribute, a.value) for a in rca_res.findings[0].attributes}
    assert ("brand", "samsung") in top_attrs
