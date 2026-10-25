"""Honest evaluation for real KPI data that has no anomaly labels.

The eCommerce behaviour dataset has no ground truth: a spike at a November 2019
promotion is not an error, and a quiet hour is not a confirmed normal hour. This
module therefore never scores detection on the untouched real series.

Instead:

* :func:`inject_anomalies` writes controlled conversion drops and traffic spikes into
  a **copy** of the real hourly KPI frame. Placement and strength are seeded and every
  window is explicitly labelled ``"injected"``.
* :func:`evaluate_injected` runs the normal detector on that copy and reports
  precision, recall and F1 against those injected labels only.
* :func:`describe_detection` reports descriptive facts (number, duration and size of
  detected windows) for the untouched series, with no performance metric.

False positives in the injected evaluation are detected windows that overlap no
injected anomaly. That is deliberately conservative: other windows may be genuine
unlabelled events in the real series, not mistakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from kpi_rca.config import AppConfig
from kpi_rca.detection import detect_anomalies
from kpi_rca.detection.kpi import preprocess_kpis
from kpi_rca.schemas import AnomalyWindow, DetectionResult

INJECTED_LABEL = "injected"
DROP = "drop"
SPIKE = "spike"
_EXPECTED_WINDOW = {DROP: "kpi_drop", SPIKE: "volume_surge"}
_REAL_DATA_NOTE = (
    "The real series has no ground-truth labels. These windows are descriptive "
    "detections, not scored incidents. Promotional peaks such as Black Friday or the "
    "November 2019 sales can appear as volume_surge and are not necessarily errors."
)


@dataclass(frozen=True)
class InjectedAnomaly:
    """One controlled anomaly written into copies of real KPI buckets."""

    anomaly_id: str
    kind: str  # "drop" (conversion) or "spike" (traffic)
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive
    n_buckets: int
    strength: float  # drop: relative conversion reduction; spike: traffic multiplier
    label: str = INJECTED_LABEL

    @property
    def expected_window(self) -> str:
        return _EXPECTED_WINDOW[self.kind]

    def as_dict(self) -> dict:
        return {
            "anomaly_id": self.anomaly_id,
            "label": self.label,
            "kind": self.kind,
            "expected_window": self.expected_window,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "n_buckets": self.n_buckets,
            "strength": float(self.strength),
        }


@dataclass
class InjectedDataset:
    """A modified copy of the real KPIs plus the labels that describe the changes."""

    kpis: pd.DataFrame
    anomalies: list[InjectedAnomaly]
    seed: int

    def labels_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.as_dict() for a in self.anomalies])


def inject_anomalies(
    kpis: pd.DataFrame,
    *,
    seed: int = 7,
    n_drops: int = 4,
    n_spikes: int = 3,
    duration_buckets: tuple[int, int] = (2, 6),
    min_gap_buckets: int = 6,
    edge_margin_buckets: int | None = None,
    drop_fraction: tuple[float, float] = (0.30, 0.60),
    spike_multiplier: tuple[float, float] = (1.6, 2.5),
) -> InjectedDataset:
    """Return a copy of ``kpis`` with seeded, labelled drops and spikes.

    Drops reduce conversion (successes and revenue fall, attempts stay); spikes raise
    traffic (attempts, successes and revenue rise, conversion stays). Only observed
    buckets with attempts are used, so imputed empty hours are never modified. The input
    frame is never mutated.
    """
    if n_drops < 0 or n_spikes < 0:
        raise ValueError("n_drops and n_spikes must be >= 0")
    if duration_buckets[0] < 1 or duration_buckets[0] > duration_buckets[1]:
        raise ValueError("duration_buckets must satisfy 1 <= low <= high")
    if min_gap_buckets < 0:
        raise ValueError("min_gap_buckets must be >= 0")
    for name, bounds in (("drop_fraction", drop_fraction), ("spike_multiplier", spike_multiplier)):
        if not (0 < bounds[0] <= bounds[1]):
            raise ValueError(f"{name} must satisfy 0 < low <= high")

    frame = preprocess_kpis(kpis).copy()
    n = len(frame)
    margin = edge_margin_buckets
    if margin is None:
        margin = max(24, 2 * _period_hint(frame))
    if n - 2 * margin <= 0:
        raise ValueError(f"series is too short for injection: {n} buckets with a {margin}-bucket edge margin")

    rng = np.random.default_rng(seed)
    durations = [
        int(rng.integers(duration_buckets[0], duration_buckets[1] + 1))
        for _ in range(n_drops + n_spikes)
    ]
    kinds = [DROP] * n_drops + [SPIKE] * n_spikes
    rng.shuffle(kinds)
    starts = _place_windows(frame, durations, margin, min_gap_buckets, rng)

    modified = frame.copy(deep=True)
    attempts = modified["attempts"].to_numpy(dtype=float, copy=True)
    successes = modified["successes"].to_numpy(dtype=float, copy=True)
    revenue = modified["revenue"].to_numpy(dtype=float, copy=True)
    conversion = modified["conversion_rate"].to_numpy(dtype=float, copy=True)
    imputed = modified["imputed"].to_numpy(dtype=bool, copy=True)

    anomalies: list[InjectedAnomaly] = []
    counters = {DROP: 0, SPIKE: 0}
    for kind, start, duration in sorted(zip(kinds, starts, durations), key=lambda item: item[1]):
        stop = start + duration
        rows = slice(start, stop)
        old_attempts = attempts[rows].copy()
        old_successes = successes[rows].copy()
        old_conversion = conversion[rows].copy()
        if kind == DROP:
            fraction = float(rng.uniform(*drop_fraction))
            new_conversion = np.clip(old_conversion * (1.0 - fraction), 0.0, 1.0)
            new_successes = np.minimum(np.rint(old_attempts * new_conversion), old_attempts)
            scale = np.divide(
                new_successes, old_successes, out=np.zeros_like(new_successes), where=old_successes > 0
            )
            new_revenue = revenue[rows] * scale
            strength = fraction
        else:
            multiplier = float(rng.uniform(*spike_multiplier))
            new_attempts = np.rint(old_attempts * multiplier)
            new_successes = np.minimum(np.rint(new_attempts * old_conversion), new_attempts)
            scale = np.divide(new_attempts, old_attempts, out=np.ones_like(new_attempts), where=old_attempts > 0)
            new_revenue = revenue[rows] * scale
            attempts[rows] = new_attempts
            strength = multiplier

        successes[rows] = new_successes
        revenue[rows] = new_revenue
        with np.errstate(divide="ignore", invalid="ignore"):
            conversion[rows] = np.where(attempts[rows] > 0, successes[rows] / attempts[rows], 0.0)
        imputed[rows] = False

        counters[kind] += 1
        prefix = "INJ-D" if kind == DROP else "INJ-S"
        anomalies.append(
            InjectedAnomaly(
                anomaly_id=f"{prefix}{counters[kind]:02d}",
                kind=kind,
                start=frame.index[start],
                end=frame.index[stop - 1] + _bucket_step(frame.index),
                n_buckets=duration,
                strength=strength,
            )
        )

    modified["attempts"] = attempts.astype(np.int64)
    modified["successes"] = successes.astype(np.int64)
    modified["failures"] = (attempts - successes).astype(np.int64)
    modified["conversion_rate"] = conversion.astype(float)
    modified["revenue"] = revenue.astype(float)
    modified["imputed"] = imputed
    anomalies.sort(key=lambda a: a.start)
    return InjectedDataset(kpis=modified, anomalies=anomalies, seed=seed)


def evaluate_injected(
    kpis: pd.DataFrame,
    cfg: AppConfig,
    *,
    seed: int = 7,
    n_drops: int = 4,
    n_spikes: int = 3,
    **injection_kwargs,
) -> dict:
    """Inject labelled anomalies into a copy of ``kpis`` and score the detector on them."""
    dataset = inject_anomalies(kpis, seed=seed, n_drops=n_drops, n_spikes=n_spikes, **injection_kwargs)
    detection = detect_anomalies(dataset.kpis, cfg.detection)
    scores = score_windows(dataset.anomalies, detection.windows)
    labels = dataset.labels_frame()
    return {
        "method": "controlled anomalies injected into a copy of the real hourly KPI series",
        "label": INJECTED_LABEL,
        "seed": seed,
        "n_injected": {
            "total": len(dataset.anomalies),
            "drop": sum(a.kind == DROP for a in dataset.anomalies),
            "spike": sum(a.kind == SPIKE for a in dataset.anomalies),
        },
        **scores,
        "injected_windows": labels.to_dict(orient="records"),
        "note": (
            "Precision and recall are measured only against the injected, labelled windows. "
            "Any detected window that does not overlap an injected anomaly counts as a false "
            "positive; this is conservative because other real events are unlabelled and may "
            "be genuine (for example the November 2019 promotional peaks)."
        ),
    }


def score_windows(
    anomalies: Sequence[InjectedAnomaly],
    windows: Sequence[AnomalyWindow],
) -> dict:
    """Greedy one-to-one overlap matching between injected labels and detected windows.

    A match requires both time overlap and the expected detector label: a conversion
    drop must produce ``kpi_drop`` and a traffic spike ``volume_surge``. Each detected
    window can match at most one injected anomaly and vice versa.
    """
    anomalies = sorted(anomalies, key=lambda a: a.start)
    used: set[int] = set()
    matches: list[dict] = []
    missed: list[dict] = []
    for anomaly in anomalies:
        best: tuple[float, int, AnomalyWindow] | None = None
        for index, window in enumerate(windows):
            if index in used or window.kpi_type != anomaly.expected_window:
                continue
            overlap = _overlap_seconds(anomaly.start, anomaly.end, window.start, window.end)
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, index, window)
        if best is None:
            missed.append(anomaly.as_dict())
            continue
        overlap, index, window = best
        used.add(index)
        matches.append(
            {
                "anomaly_id": anomaly.anomaly_id,
                "kind": anomaly.kind,
                "window_id": window.window_id,
                "window_kpi_type": window.kpi_type,
                "overlap_hours": overlap / 3600.0,
            }
        )

    true_positives = len(matches)
    false_negatives = len(missed)
    false_positives = len(windows) - true_positives
    precision = true_positives / len(windows) if windows else None
    recall = true_positives / len(anomalies) if anomalies else None
    overall = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }

    by_kind: dict[str, dict] = {}
    for kind in (DROP, SPIKE):
        expected = _EXPECTED_WINDOW[kind]
        n_labels = sum(a.kind == kind for a in anomalies)
        n_windows = sum(w.kpi_type == expected for w in windows)
        n_matched = sum(m["kind"] == kind for m in matches)
        kind_precision = n_matched / n_windows if n_windows else None
        kind_recall = n_matched / n_labels if n_labels else None
        by_kind[kind] = {
            "n_injected": n_labels,
            "n_detected_windows": n_windows,
            "true_positives": n_matched,
            "precision": kind_precision,
            "recall": kind_recall,
            "f1": _f1(kind_precision, kind_recall),
        }

    return {
        **overall,
        "by_kind": by_kind,
        "matches": matches,
        "missed_injected": missed,
        "unmatched_detected": [
            _window_record(w) for index, w in enumerate(windows) if index not in used
        ],
    }


def describe_detection(detection: DetectionResult) -> dict:
    """Descriptive facts about detections on the real series; no performance metric."""
    windows = detection.windows
    drops = [w for w in windows if w.kpi_type == "kpi_drop"]
    surges = [w for w in windows if w.kpi_type == "volume_surge"]
    return {
        "n_buckets": int(len(detection.scores)),
        "first_bucket": detection.scores.index[0].isoformat() if len(detection.scores) else None,
        "last_bucket": detection.scores.index[-1].isoformat() if len(detection.scores) else None,
        "n_windows": len(windows),
        "n_kpi_drop": len(drops),
        "n_volume_surge": len(surges),
        "kpi_drop_hours": float(sum(w.duration_hours for w in drops)),
        "volume_surge_hours": float(sum(w.duration_hours for w in surges)),
        "windows": [_window_record(w) for w in windows],
        "note": _REAL_DATA_NOTE,
    }


# ------------------------------------------------------------------ internals


def _place_windows(
    frame: pd.DataFrame,
    durations: list[int],
    margin: int,
    min_gap: int,
    rng: np.random.Generator,
) -> list[int]:
    observed = frame["attempts"].to_numpy(dtype=float) > 0
    n = len(frame)
    lo = margin
    hi = n - margin

    def usable(start: int, duration: int) -> bool:
        return start >= lo and start + duration <= hi and bool(observed[start : start + duration].all())

    for _ in range(5000):
        starts: list[int] = []
        spans: list[tuple[int, int]] = []
        for duration in durations:
            if hi - lo < duration:
                break
            start = int(rng.integers(lo, hi - duration + 1))
            if not usable(start, duration):
                break
            stop = start + duration
            if any(not (stop + min_gap <= b or start >= e + min_gap) for b, e in spans):
                break
            starts.append(start)
            spans.append((start, stop))
        if len(starts) == len(durations):
            return starts
    raise ValueError(
        "could not place the injected anomalies without overlap; use more observed buckets, "
        "fewer anomalies or shorter durations"
    )


def _period_hint(frame: pd.DataFrame) -> int:
    if isinstance(frame.index, pd.DatetimeIndex):
        try:
            step = _bucket_step(frame.index)
        except ValueError:
            return 24
        if step == pd.Timedelta(hours=1):
            return 24
        if step == pd.Timedelta(days=1):
            return 7
    return 24


def _bucket_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("need at least two buckets to infer the bucket size")
    return index[1] - index[0]


def _overlap_seconds(a_start, a_end, b_start, b_end) -> float:
    start = max(pd.Timestamp(a_start), pd.Timestamp(b_start))
    end = min(pd.Timestamp(a_end), pd.Timestamp(b_end))
    return max(0.0, (end - start).total_seconds())


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _window_record(window: AnomalyWindow) -> dict:
    return {
        "window_id": window.window_id,
        "kpi_type": window.kpi_type,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "n_buckets": window.n_buckets,
        "duration_hours": window.duration_hours,
        "drop_pct": window.drop_pct,
        "peak_anomaly_score": window.peak_anomaly_score,
        "reason": window.reason,
    }
