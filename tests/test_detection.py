"""Detection: KPI aggregation, batch STL + Isolation Forest windows, labels and the streaming replay."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from conftest import SMALL_INCIDENT_START, SMALL_SURGE_START
from kpi_rca.config import DetectionConfig
from kpi_rca.detection import StreamingDetector, aggregate_kpis, detect_anomalies, detection_delays
from kpi_rca.detection.detector import (
    Z_COLUMNS,
    direction_labels,
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
