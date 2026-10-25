"""Detection: KPI aggregation, batch STL + Isolation Forest windows, labels and the streaming replay."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from conftest import SMALL_INCIDENT_START, SMALL_SURGE_START
from kpi_rca.config import DetectionConfig
from kpi_rca.detection import StreamingDetector, aggregate_kpis, detect_anomalies, detection_delays, preprocess_kpis
from kpi_rca.detection.detector import (
    InsufficientDataError,
    InvalidKPIError,
    Z_COLUMNS,
    combine_signals,
    direction_labels,
    fit_isolation_forest,
    group_windows,
    robust_z,
    uses_mstl,
)
from kpi_rca.schemas import KPI_COLUMNS, AnomalyWindow

INCIDENT_END = SMALL_INCIDENT_START + timedelta(hours=4)
SURGE_END = SMALL_SURGE_START + timedelta(hours=5)


def _overlaps(w: AnomalyWindow, start: datetime, end: datetime) -> bool:
    return w.start < end and w.end > start


@pytest.fixture(scope="module")
def small_kpis(small_scenario):
    return aggregate_kpis(small_scenario.transactions)


@pytest.fixture(scope="module")
def small_detection(small_kpis, small_cfg):
    return detect_anomalies(small_kpis, small_cfg.detection)


# ------------------------------------------------------------------ aggregation


def _tiny_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-09-01 00:10", "2026-09-01 00:50", "2026-09-01 02:30"]),
            "success": [1, 0, 1],
            "amount": [10.0, 20.0, 30.0],
            "latency_ms": [100.0, 300.0, 200.0],
        }
    )


def test_aggregate_kpis_sums_and_fills_empty_buckets():
    kpis = aggregate_kpis(_tiny_transactions())
    assert list(kpis.index) == list(pd.date_range("2026-09-01 00:00", periods=3, freq="h"))
    assert kpis.index.name == "bucket"
    assert kpis["attempts"].tolist() == [2, 0, 1]
    assert kpis["failures"].tolist() == [1, 0, 0]
    assert kpis["revenue"].tolist() == [10.0, 0.0, 30.0]  # only successful amounts count
    # the empty 01:00 bucket gets an interpolated conversion rate, not NaN
    assert kpis["conversion_rate"].tolist() == [0.5, 0.75, 1.0]


def test_aggregate_kpis_explicit_range_and_outside_rows():
    kpis = aggregate_kpis(_tiny_transactions(), start=datetime(2026, 9, 1), end=datetime(2026, 9, 1, 5))
    assert len(kpis) == 5 and kpis["attempts"].sum() == 3
    with pytest.raises(ValueError, match="outside"):
        aggregate_kpis(_tiny_transactions(), start=datetime(2026, 9, 1, 1), end=datetime(2026, 9, 1, 5))


def test_aggregate_kpis_matches_transactions(small_scenario, small_kpis):
    tx = small_scenario.transactions
    assert small_kpis["attempts"].sum() == len(tx)
    assert small_kpis["successes"].sum() == tx["success"].sum()
    assert small_kpis.index.is_monotonic_increasing
    assert (small_kpis.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


# ------------------------------------------------------------------ batch


def test_batch_scores_have_the_spec_columns(small_detection, small_kpis):
    scores = small_detection.scores
    assert len(scores) == len(small_kpis)
    for kpi in KPI_COLUMNS:
        assert f"{kpi}_expected" in scores and f"{kpi}_resid_z" in scores
    assert set(scores["label"].unique()) <= {"normal", "kpi_drop", "volume_surge"}
    assert scores["is_anomaly"].dtype == bool
    # an unflagged bucket is always "normal"
    assert (scores.loc[~scores["is_anomaly"], "label"] == "normal").all()


def test_batch_detects_the_incident_as_a_drop(small_detection):
    drops = [w for w in small_detection.windows if w.kpi_type == "kpi_drop"]
    hits = [w for w in drops if _overlaps(w, SMALL_INCIDENT_START, INCIDENT_END)]
    assert hits, f"no kpi_drop window overlaps the incident; windows: {small_detection.windows}"
    w = hits[0]
    assert w.observed["conversion_rate"] < w.expected["conversion_rate"]
    assert set(w.observed) == set(KPI_COLUMNS) == set(w.expected)
    assert w.detected_at is None  # batch mode never sets it


def test_batch_never_calls_the_surge_a_drop(small_detection):
    assert not [
        w for w in small_detection.windows if w.kpi_type == "kpi_drop" and _overlaps(w, SMALL_SURGE_START, SURGE_END)
    ]
    assert [w for w in small_detection.windows if w.kpi_type == "volume_surge" and _overlaps(w, SMALL_SURGE_START, SURGE_END)]


def test_windows_are_ordered_with_exclusive_ends(small_detection):
    windows = small_detection.windows
    assert [w.window_id for w in windows] == [f"W-{i:03d}" for i in range(1, len(windows) + 1)]
    for w in windows:
        assert w.end == w.start + timedelta(hours=w.n_buckets)
        assert w.n_buckets >= 2  # min_window_buckets debounce
    assert all(a.end <= b.start for a, b in zip(windows, windows[1:]))


# ------------------------------------------------------------- building blocks


def test_direction_labels():
    signals = pd.DataFrame(
        {
            "conversion_rate_resid_z": [-6.0, 0.5, -1.0, 2.0, -6.0],
            "revenue_resid_z": [-5.0, 8.0, 1.0, 1.0, -5.0],
            "attempts_resid_z": [0.0, 9.0, 4.0, 0.0, 0.0],
        }
    )[Z_COLUMNS]
    flagged = np.array([True, True, True, True, False])
    labels = direction_labels(signals, flagged).tolist()
    # drop; surge; surge (attempts up more than conversion down); nothing down -> normal; unflagged
    assert labels == ["kpi_drop", "volume_surge", "volume_surge", "normal", "normal"]


def test_group_windows_debounce_gap_and_label_change():
    n, d, s = "normal", "kpi_drop", "volume_surge"
    labels = np.array([d, n, n, d, d, n, d, n, n, s, s, d, d])
    spans = group_windows(labels, gap_tolerance=1, min_buckets=2)
    # the lone drop at 0 is debounced away; 3..6 is one window across a 1-bucket gap;
    # the surge and the drop right after it are separate windows
    assert [(sp.kind, sp.first, sp.last) for sp in spans] == [(d, 3, 6), (s, 9, 10), (d, 11, 12)]
    assert spans[0].confirmed_at == 4


def test_robust_z_resists_outliers_and_constant_input():
    values = np.r_[np.random.default_rng(0).normal(0, 1, 500), [50.0]]
    z = robust_z(values)
    assert z[-1] > 30  # the outlier does not inflate the scale
    assert np.isfinite(robust_z(np.ones(10))).all()


def test_mstl_switches_on_after_the_configured_weeks():
    cfg = DetectionConfig()
    assert not uses_mstl(21 * 24, cfg)
    assert uses_mstl(int(cfg.mstl_min_weeks * 168), cfg)


# ------------------------------------------------------------------ streaming


def test_streaming_detects_the_incident_with_a_delay(small_kpis, small_cfg):
    detector = StreamingDetector(small_cfg.detection)
    alerts = []
    result = detector.replay(
        small_kpis,
        start=SMALL_INCIDENT_START - timedelta(hours=3),
        end=INCIDENT_END + timedelta(hours=2),
        on_alert=alerts.append,
    )
    assert result.steps == 9
    drops = [a for a in alerts if a.kpi_type == "kpi_drop"]
    assert drops, f"no streaming alert; scores:\n{result.scores[['label', 'if_score']]}"
    assert [w.window_id for w in result.windows] == sorted(w.window_id for w in result.windows)

    [delay] = detection_delays(result.windows, [_Incident("INC-T1", SMALL_INCIDENT_START, INCIDENT_END)])
    assert delay.detected and delay.detected_at is not None
    # the 2-bucket debounce sets a floor of 2 h; a good detector confirms within the incident
    assert 2.0 <= delay.delay_h <= 4.0


def test_streaming_never_looks_ahead(small_kpis, small_cfg):
    """The score of bucket t must not change when everything after t is changed."""
    detector = StreamingDetector(small_cfg.detection)
    t = SMALL_INCIDENT_START + timedelta(hours=1)
    altered = small_kpis.copy()
    future = altered.index > t
    altered.loc[future, "attempts"] = altered.loc[future, "attempts"] * 3
    altered.loc[future, "conversion_rate"] = 0.2
    a = detector.replay(small_kpis, start=t, end=t + timedelta(hours=1)).scores
    b = detector.replay(altered, start=t, end=t + timedelta(hours=1)).scores
    pd.testing.assert_frame_equal(a, b)


def test_partial_replay_scores_match_a_longer_replay(small_kpis, small_cfg):
    """The refit grid makes scores independent of where the replay starts."""
    detector = StreamingDetector(small_cfg.detection)
    t = SMALL_INCIDENT_START
    short = detector.replay(small_kpis, start=t, end=t + timedelta(hours=3)).scores
    longer = detector.replay(small_kpis, start=t - timedelta(hours=4), end=t + timedelta(hours=3)).scores
    cols = ["if_score", *Z_COLUMNS]
    pd.testing.assert_frame_equal(short[cols], longer.loc[short.index, cols])


def test_streaming_needs_a_warmup(small_kpis, small_cfg):
    detector = StreamingDetector(small_cfg.detection)
    with pytest.raises(ValueError, match="streaming needs more than"):
        detector.replay(small_kpis.iloc[: detector.warmup_buckets])
    with pytest.raises(ValueError, match="warm-up"):
        StreamingDetector(DetectionConfig(stream_warmup_days=1.5, stream_history_days=2))


class _Incident:
    def __init__(self, incident_id: str, start: datetime, end: datetime) -> None:
        self.incident_id, self.start, self.end = incident_id, start, end


def test_detection_delays_matching():
    def window(wid, start_h, n, kind="kpi_drop", detected_h=None):
        start = datetime(2026, 9, 1) + timedelta(hours=start_h)
        return AnomalyWindow(
            window_id=wid,
            start=start,
            end=start + timedelta(hours=n),
            kpi_type=kind,
            n_buckets=n,
            if_score_min=-0.1,
            observed={},
            expected={},
            detected_at=None if detected_h is None else datetime(2026, 9, 1) + timedelta(hours=detected_h),
        )

    incidents = [
        _Incident("A", datetime(2026, 9, 1, 10), datetime(2026, 9, 1, 14)),
        _Incident("B", datetime(2026, 9, 2, 10), datetime(2026, 9, 2, 12)),
    ]
    windows = [window("S-1", 11, 3, detected_h=13), window("S-2", 34, 2, kind="volume_surge", detected_h=36)]
    a, b = detection_delays(windows, incidents)
    assert (a.detected, a.window_id, a.delay_h) == (True, "S-1", 3.0)
    assert (b.detected, b.delay_h) == (False, None)  # a surge window never counts as detection


# ------------------------------------------------- preprocessing and validation


def test_aggregate_kpis_marks_imputed_buckets():
    kpis = aggregate_kpis(_tiny_transactions())
    # only the empty 01:00 bucket needed interpolation (conversion and latency)
    assert kpis["imputed"].tolist() == [False, True, False]


def test_aggregate_kpis_merges_duplicate_rows_by_bucket():
    tx = _tiny_transactions()
    extra = tx.iloc[[1]].copy()
    extra["success"], extra["amount"] = 1, 5.0  # a second attempt in the same bucket
    kpis = aggregate_kpis(pd.concat([tx, extra]))
    # documented rule: counts and revenue sum, conversion recomputes on the merged bucket
    assert kpis["attempts"].tolist()[0] == 3 and kpis["successes"].tolist()[0] == 2
    assert kpis["revenue"].tolist()[0] == 15.0
    assert kpis["conversion_rate"].tolist()[0] == pytest.approx(2 / 3)


def test_preprocess_kpis_merges_duplicate_buckets_and_sorts():
    index = pd.DatetimeIndex(
        ["2026-09-01 01:00", "2026-09-01 00:00", "2026-09-01 00:00"], name="bucket"
    )
    frame = pd.DataFrame(
        {
            "attempts": [1, 2, 3],
            "successes": [1, 2, 1],
            "conversion_rate": [1.0, 1.0, 1.0 / 3.0],
            "revenue": [10.0, 20.0, 5.0],
            "avg_latency_ms": [100.0, 200.0, 400.0],
            "imputed": [False, False, False],
        },
        index=index,
    )
    out = preprocess_kpis(frame)
    assert list(out.index) == list(pd.date_range("2026-09-01 00:00", periods=2, freq="h"))
    assert out["attempts"].tolist() == [5, 1]
    assert out["successes"].tolist() == [3, 1]
    assert out["failures"].tolist() == [2, 0]
    # conversion is recomputed from the merged counts, not averaged from the rates
    assert out["conversion_rate"].tolist() == [0.6, 1.0]
    assert out["revenue"].tolist() == [25.0, 10.0]
    assert out["avg_latency_ms"].tolist() == [300.0, 100.0]


def test_preprocess_kpis_interpolates_and_flags_missing_rates():
    index = pd.date_range("2026-09-01", periods=3, freq="h", name="bucket")
    frame = pd.DataFrame(
        {
            "attempts": [10, 0, 10],
            "successes": [9, 0, 10],
            "conversion_rate": [0.9, np.nan, 1.0],
            "revenue": [10.0, 0.0, 20.0],
            "avg_latency_ms": [100.0, np.nan, 200.0],
        },
        index=index,
    )
    out = preprocess_kpis(frame)
    assert out["conversion_rate"].tolist() == [0.9, 0.95, 1.0]
    assert out["avg_latency_ms"].tolist() == [100.0, 150.0, 200.0]
    assert out["imputed"].tolist() == [False, True, False]
    assert out["attempts"].tolist() == [10, 0, 10]  # counts stay 0, they are not invented


def test_preprocess_kpis_rejects_impossible_values():
    index = pd.date_range("2026-09-01", periods=3, freq="h", name="bucket")
    base = pd.DataFrame(
        {
            "attempts": [10, 10, 10],
            "successes": [9, 9, 9],
            "conversion_rate": [0.9, 0.9, 0.9],
            "revenue": [10.0, 10.0, 10.0],
            "avg_latency_ms": [100.0, 100.0, 100.0],
        },
        index=index,
    )
    with pytest.raises(InvalidKPIError, match="conversion_rate values outside"):
        bad = base.copy()
        bad["conversion_rate"] = [0.9, 1.5, 0.9]
        preprocess_kpis(bad)
    with pytest.raises(InvalidKPIError, match="negative"):
        bad = base.copy()
        bad["revenue"] = [10.0, -1.0, 10.0]
        preprocess_kpis(bad)
    with pytest.raises(InvalidKPIError, match="NaN or infinite"):
        bad = base.copy()
        bad["revenue"] = [10.0, np.inf, 10.0]
        preprocess_kpis(bad)
    with pytest.raises(InvalidKPIError, match="missing columns"):
        preprocess_kpis(base.drop(columns="conversion_rate"))
    with pytest.raises(InvalidKPIError, match="DatetimeIndex"):
        preprocess_kpis(base.reset_index(drop=True))


def _kpi_frame(rng: np.random.Generator, hours: int = 120) -> pd.DataFrame:
    """A clean hourly KPI series: attempts like traffic, sensible conversion and revenue."""
    index = pd.date_range("2026-09-01", periods=hours, freq="h", name="bucket")
    attempts = rng.poisson(500 + 50 * np.sin(2 * np.pi * index.hour / 24))
    conversion = np.clip(0.86 + 0.04 * np.cos(2 * np.pi * (index.hour - 15) / 24) + rng.normal(0, 0.004, hours), 0, 1)
    successes = np.round(attempts * conversion).astype(int)
    revenue = successes * 40.0 * (1 + rng.normal(0, 0.02, hours))
    return pd.DataFrame(
        {
            "attempts": attempts,
            "successes": successes,
            "failures": attempts - successes,
            "conversion_rate": conversion,
            "revenue": revenue,
            "avg_latency_ms": np.full(hours, 420.0),
        },
        index=index,
    )


def test_detection_interpolates_missing_values_and_marks_them_imputed():
    rng = np.random.default_rng(7)
    kpis = _kpi_frame(rng)
    neighbours = (kpis["conversion_rate"].iloc[19] + kpis["conversion_rate"].iloc[22]) / 2
    kpis.iloc[20:22, kpis.columns == "conversion_rate"] = np.nan
    det = detect_anomalies(kpis, DetectionConfig())
    filled = det.scores.loc[det.scores["imputed"], "conversion_rate"]
    assert list(filled.index) == [kpis.index[20], kpis.index[21]]
    # the interpolated values sit between their neighbours: no invention, no exception
    assert min(filled) > neighbours - 0.01 and max(filled) < neighbours + 0.01
    assert np.isfinite(det.scores["conversion_rate_resid_z"]).all()


def test_edge_gap_in_a_kpi_frame_is_still_scored():
    rng = np.random.default_rng(8)
    kpis = _kpi_frame(rng)
    kpis.iloc[:2, kpis.columns == "conversion_rate"] = np.nan  # edge gap: nearest value carried back
    det = detect_anomalies(kpis, DetectionConfig())
    assert det.scores["imputed"].head(2).all()
    assert len(det.scores) == 120


def test_insufficient_data_raises_a_typed_descriptive_error():
    rng = np.random.default_rng(9)
    kpis = _kpi_frame(rng, hours=30)
    with pytest.raises(InsufficientDataError, match="two full seasonal periods.*got 30"):
        detect_anomalies(kpis, DetectionConfig())


def test_constant_series_is_never_anomalous():
    index = pd.date_range("2026-09-01", periods=60, freq="h", name="bucket")
    kpis = pd.DataFrame(
        {
            "attempts": np.full(60, 500, dtype=np.int64),
            "successes": np.full(60, 430, dtype=np.int64),
            "failures": np.full(60, 70, dtype=np.int64),
            "conversion_rate": np.full(60, 0.86),
            "revenue": np.full(60, 17200.0),
            "avg_latency_ms": np.full(60, 420.0),
        },
        index=index,
    )
    det = detect_anomalies(kpis, DetectionConfig())
    assert not det.scores["is_anomaly"].any()
    assert det.windows == []
    assert np.isfinite(det.scores["anomaly_score"]).all()


def test_detection_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(11)
    kpis = _kpi_frame(rng)
    first = detect_anomalies(kpis, DetectionConfig())
    second = detect_anomalies(kpis, DetectionConfig())
    pd.testing.assert_frame_equal(first.scores, second.scores)
    assert first.windows == second.windows


def test_anomaly_score_is_higher_for_more_anomalous():
    rng = np.random.default_rng(12)
    kpis = _kpi_frame(rng)
    kpis.iloc[60:64, kpis.columns == "conversion_rate"] = 0.2  # a 4-hour drop
    det = detect_anomalies(kpis, DetectionConfig())
    scores = det.scores
    assert (scores["anomaly_score"] == -scores["if_score"]).all()
    flagged = scores["anomaly_score"][scores["is_anomaly"]]
    normal = scores["anomaly_score"][~scores["is_anomaly"]]
    assert flagged.min() > normal.quantile(0.5)
    assert flagged.max() == scores["anomaly_score"].max()


def test_flagged_points_carry_a_human_readable_reason():
    rng = np.random.default_rng(13)
    kpis = _kpi_frame(rng)
    kpis.iloc[60:64, kpis.columns == "conversion_rate"] = 0.5
    det = detect_anomalies(kpis, DetectionConfig())
    scores = det.scores
    assert (scores.loc[~scores["is_anomaly"], "reason"] == "").all()
    flagged = scores.loc[scores["is_anomaly"], "reason"]
    assert flagged.str.contains("MAD").all() and flagged.str.contains("IF score").all()
    sample = flagged.iloc[0]
    assert "conversion_rate" in sample and "seasonal expectation" in sample


def test_combine_signals_agrees_or_requires_one_extreme():
    cfg = DetectionConfig()  # min_gate_z 3.0, extreme_z 6.0
    index = pd.date_range("2026-09-01", periods=4, freq="h", name="bucket")
    z = pd.DataFrame(
        {"conversion_rate_resid_z": [0.1, -4.0, -0.5, 7.0], "revenue_resid_z": 0.0, "attempts_resid_z": 0.0},
        index=index,
    )
    if_flagged = np.array([True, True, False, False])
    # forest flags AND |z| >= 3 -> agree; forest flags but |z| < 3 -> gated away;
    # |z| >= 6 without the forest -> the extreme clause keeps it
    assert combine_signals(if_flagged, z, cfg).tolist() == [False, True, False, True]


def test_contamination_is_configurable():
    cfg = DetectionConfig(contamination=0.05)
    rng = np.random.default_rng(14)
    model = fit_isolation_forest(rng.normal(size=(300, 3 * 2)), cfg)
    assert model.contamination == 0.05  # default is "auto"; a number passes through to sklearn
    assert model.n_estimators == cfg.n_estimators and model.random_state == cfg.random_state
    det = detect_anomalies(_kpi_frame(np.random.default_rng(15)), cfg)
    assert det.scores["is_anomaly"].dtype == bool  # scores with a fixed contamination are usable


def test_period_is_inferred_from_the_frequency():
    assert DetectionConfig().period == 24
    assert DetectionConfig(freq="1D").period == 7
    assert DetectionConfig(freq="15min").period == 96
    with pytest.raises(ValueError, match="cannot infer"):
        DetectionConfig(freq="W")


def test_weekly_period_defaults_to_seven_daily_periods():
    assert DetectionConfig().weekly_period == 168
    assert DetectionConfig(freq="1D").weekly_period == 49
    assert DetectionConfig(period=24, weekly_period=168).weekly_period == 168


def test_window_summary_fields(small_detection):
    drops = [w for w in small_detection.windows if w.kpi_type == "kpi_drop"]
    assert drops and all(w.reason for w in small_detection.windows)
    assert all(w.duration_hours == w.n_buckets for w in small_detection.windows)
    assert all(w.peak_anomaly_score > 0 for w in small_detection.windows)
    assert all(w.drop_pct is not None and w.drop_pct > 0 for w in drops)  # below seasonal expectation
    scores = small_detection.scores
    for w in small_detection.windows:
        rows = scores.loc[(scores.index >= w.start) & (scores.index < w.end)]
        assert w.peak_anomaly_score == pytest.approx(rows["anomaly_score"].max())
