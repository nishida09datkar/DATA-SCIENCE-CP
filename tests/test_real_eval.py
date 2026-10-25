"""Real-data honesty: injected evaluation and the descriptive demo."""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

from kpi_rca.cli import main
from kpi_rca.config import AppConfig
from kpi_rca.demo import load_hourly_kpis, run_demo
from kpi_rca.detection import detect_anomalies
from kpi_rca.injected_eval import (
    INJECTED_LABEL,
    InjectedAnomaly,
    describe_detection,
    evaluate_injected,
    inject_anomalies,
    score_windows,
)
from kpi_rca.schemas import AnomalyWindow


def _hourly_frame(hours: int = 24 * 12) -> pd.DataFrame:
    """A realistic-looking hourly KPI table with daily seasonality and mild noise."""
    rng = np.random.default_rng(21)
    index = pd.date_range("2019-10-01", periods=hours, freq="h", name="hour")
    hour = index.hour.to_numpy()
    attempts = np.rint(
        900 + 350 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 25, hours)
    ).astype(np.int64)
    attempts = np.maximum(attempts, 50)
    conversion = np.clip(
        0.80 + 0.04 * np.cos(2 * np.pi * (hour - 15) / 24) + rng.normal(0, 0.006, hours), 0.4, 0.95
    )
    successes = np.minimum(np.rint(attempts * conversion).astype(np.int64), attempts)
    revenue = successes * 45.0 * (1 + rng.normal(0, 0.03, hours))
    return pd.DataFrame(
        {
            "attempts": attempts,
            "successes": successes,
            "failures": attempts - successes,
            "conversion_rate": successes / attempts,
            "revenue": revenue,
            "imputed": False,
        },
        index=index,
    )


def _window(window_id: str, start: pd.Timestamp, end: pd.Timestamp, kind: str) -> AnomalyWindow:
    return AnomalyWindow(
        window_id=window_id,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        kpi_type=kind,
        n_buckets=int((end - start) / pd.Timedelta(hours=1)),
        if_score_min=-0.2,
        observed={},
        expected={},
    )


def test_injection_is_seeded_labelled_and_never_mutates_the_input():
    original = _hourly_frame()
    untouched = original.copy(deep=True)

    first = inject_anomalies(original, seed=5, n_drops=3, n_spikes=2)
    second = inject_anomalies(original, seed=5, n_drops=3, n_spikes=2)

    pd.testing.assert_frame_equal(original, untouched)  # the real series is only copied
    pd.testing.assert_frame_equal(first.kpis, second.kpis)
    assert [a.as_dict() for a in first.anomalies] == [a.as_dict() for a in second.anomalies]
    assert all(a.label == INJECTED_LABEL for a in first.anomalies)
    assert sum(a.kind == "drop" for a in first.anomalies) == 3
    assert sum(a.kind == "spike" for a in first.anomalies) == 2


def test_injected_drops_and_spikes_move_only_their_windows():
    original = _hourly_frame()
    dataset = inject_anomalies(original, seed=9, n_drops=3, n_spikes=2)
    for anomaly in dataset.anomalies:
        rows = (dataset.kpis.index >= anomaly.start) & (dataset.kpis.index < anomaly.end)
        if anomaly.kind == "drop":
            assert (
                dataset.kpis.loc[rows, "conversion_rate"]
                < original.loc[rows, "conversion_rate"]
            ).all()
            assert (dataset.kpis.loc[rows, "attempts"] == original.loc[rows, "attempts"]).all()
        else:
            assert (dataset.kpis.loc[rows, "attempts"] > original.loc[rows, "attempts"]).all()

    injected = dataset.labels_frame()
    assert set(injected["label"]) == {INJECTED_LABEL}
    assert set(injected["expected_window"]) == {"kpi_drop", "volume_surge"}


def test_score_windows_matches_kind_and_time_one_to_one():
    start = pd.Timestamp("2019-10-03 10:00")
    labels = [
        InjectedAnomaly("INJ-D01", "drop", start, start + timedelta(hours=4), 4, 0.5),
        InjectedAnomaly("INJ-S01", "spike", start + timedelta(days=1), start + timedelta(days=1, hours=3), 3, 2.0),
    ]
    windows = [
        _window("W-001", start + timedelta(hours=1), start + timedelta(hours=5), "kpi_drop"),
        _window("W-002", start + timedelta(days=1), start + timedelta(days=1, hours=2), "kpi_drop"),
        _window("W-003", start + timedelta(days=3), start + timedelta(days=3, hours=2), "volume_surge"),
    ]
    scores = score_windows(labels, windows)
    # the spike overlaps a window with the wrong label, so it is missed; the unrelated
    # surplus window is a false positive
    assert (scores["true_positives"], scores["false_positives"], scores["false_negatives"]) == (1, 2, 1)
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["recall"] == pytest.approx(1 / 2)
    assert scores["by_kind"]["drop"]["recall"] == pytest.approx(1.0)
    assert scores["by_kind"]["spike"]["recall"] == pytest.approx(0.0)


def test_evaluate_injected_scores_only_injected_labels():
    metrics = evaluate_injected(_hourly_frame(), AppConfig(), seed=7, n_drops=3, n_spikes=2)
    assert metrics["label"] == INJECTED_LABEL
    assert metrics["n_injected"] == {"total": 5, "drop": 3, "spike": 2}
    assert metrics["seed"] == 7
    assert len(metrics["injected_windows"]) == 5
    assert metrics["true_positives"] + metrics["false_negatives"] == 5
    for key in ("precision", "recall", "f1"):
        assert metrics[key] is None or 0.0 <= metrics[key] <= 1.0
    assert set(metrics["by_kind"]) == {"drop", "spike"}
    assert "false positive" in metrics["note"]


def test_descriptive_results_are_not_performance_metrics():
    description = describe_detection(detect_anomalies(_hourly_frame(), AppConfig().detection))
    assert description["n_buckets"] == 24 * 12
    assert description["n_windows"] == description["n_kpi_drop"] + description["n_volume_surge"]
    assert "precision" not in description and "recall" not in description
    assert "no ground-truth labels" in description["note"]
    for window in description["windows"]:
        assert {"window_id", "kpi_type", "start", "end", "n_buckets", "reason"} <= set(window)


def test_demo_saves_plot_and_summary_and_can_run_injected_evaluation(tmp_path):
    kpi_path = tmp_path / "hourly_kpis.csv"
    _hourly_frame().to_csv(kpi_path, index_label="hour")

    result = run_demo(
        kpi_path,
        tmp_path / "demo",
        AppConfig(),
        evaluate_injected_anomalies=True,
        seed=4,
        n_drops=2,
        n_spikes=1,
    )
    assert result.plot_path.is_file() and result.plot_path.stat().st_size > 0
    assert result.windows_path.is_file() and result.summary_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["descriptive"]["n_buckets"] == 24 * 12
    assert summary["injected"]["label"] == INJECTED_LABEL


def test_load_hourly_kpis_accepts_an_aggregated_table(tmp_path):
    path = tmp_path / "kpis.csv"
    _hourly_frame().to_csv(path, index_label="bucket")
    kpis = load_hourly_kpis(path)
    assert len(kpis) == 24 * 12 and kpis.index.name == "bucket"
    assert {"attempts", "successes", "conversion_rate", "revenue"} <= set(kpis.columns)


def test_demo_cli_prints_windows_and_writes_outputs(tmp_path, capsys):
    kpi_path = tmp_path / "kpis.csv"
    _hourly_frame().to_csv(kpi_path, index_label="hour")
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"detection": {"period": 24}}), encoding="utf-8")

    code = main(
        ["demo", "--config", str(config), "--input", str(kpi_path), "--out-dir", str(tmp_path / "demo")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "hourly buckets" in out and "window(s)" in out
    assert "no ground-truth labels" in out
    assert (tmp_path / "demo" / "detection.png").is_file()
