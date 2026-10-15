"""Streaming replay: score KPI buckets one at a time, as they would arrive live.

When a new bucket arrives, the detector sees only the trailing ``stream_history_days``
of KPI history, never anything later:

1. Expected value = a one-step-ahead forecast from the buckets BEFORE the new one. STL is
   fitted on that past; the forecast is the same hour's seasonal value one period ago
   plus the median deseasonalised level of the last ``stream_level_buckets`` buckets
   (default 25, the STL trend window: about a day).
   Why not fit STL on the history including the new bucket and read off its last value?
   At the end of a series the loess smoothers are one-sided, so the fit bends towards
   the newest point and absorbs most of an ongoing drop (measured on the test scenario:
   expected 0.837 vs observed 0.835 inside a batch z = -7.7 incident, so nothing was
   flagged). The median level also ignores the first incident buckets once they are in
   the past, where a loess trend would already be pulled down by them.
2. z-score of the new bucket: its variance-stabilised forecast residual, robust-scaled by
   the median and MAD of the same forecast rule's errors on the past buckets
   (``robust_z`` with a reference), so noise has |z| ~ 1 as in batch. The rolling-mean
   features use the live z-scores of the previous buckets, computed the same way when
   those buckets arrived.
3. The Isolation Forest is refit every ``stream_refit_every`` buckets on the history's
   in-sample features (as in batch); in between, the last fitted forest scores the
   newest bucket. Refits sit on a fixed grid (counted from the end of the warm-up), so
   replaying only part of a series gives the same scores as a full replay.
   Known limitation: a live forecast cannot know today's traffic level at the start of
   the day, so day-level shifts (weekday profile, daily noise) give more false windows
   than batch detection (default scenario: 2-3 false kpi_drop windows in 14 replayed
   days vs none in batch on seed 42). Streaming measures detection delay; the reports
   come from batch detection.
4. Labels go through the same WindowTracker as batch detection. An alert fires the
   moment a window is confirmed (``min_window_buckets`` flagged buckets). It fires at
   the end of the confirming bucket, so detection delay = detected_at - incident start.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.ensemble import IsolationForest

from kpi_rca.config import DetectionConfig
from kpi_rca.detection.detector import (
    EXPECTED_COLUMNS,
    KPI_COLUMNS,
    Z_COLUMNS,
    Span,
    WindowTracker,
    _check_kpis,
    decompose,
    direction_labels,
    expected_kpis,
    fit_isolation_forest,
    if_features,
    make_window,
    residual_zscores,
    robust_z,
    stabilised_residuals,
    uses_mstl,
    window_kpis,
)
from kpi_rca.schemas import AnomalyWindow

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamAlert:
    """What a live system knows when a window is confirmed (its end is not known yet)."""

    window_id: str
    kpi_type: str
    start: datetime  # first flagged bucket
    detected_at: datetime  # end of the bucket that confirmed the window
    observed: dict[str, float]  # KPI totals over [start, detected_at)
    expected: dict[str, float]


@dataclass
class StreamResult:
    alerts: list[StreamAlert]
    windows: list[AnomalyWindow]  # closed windows; detected_at = when the alert fired
    scores: pd.DataFrame  # live per-bucket scores, as computed when each bucket arrived
    steps: int
    runtime_s: float


def bucket_step(freq: str) -> pd.Timedelta:
    try:
        return pd.Timedelta(to_offset(freq))
    except ValueError as exc:
        raise ValueError(f"detection.freq {freq!r} must be a fixed-length offset for streaming") from exc


@dataclass
class StreamingDetector:
    """Replays a KPI frame bucket by bucket (see the module docstring).

    Keep stream_warmup_days at about 5 or more. The forest trains on in-sample STL
    residuals, and on 3 days STL fits almost every bucket exactly: the residual MAD is
    then near zero, the training z-scores reach the thousands, and a live z of -7 looks
    normal (seen at 600 attempts/h with a 3-day warm-up: no alert for any incident).
    """

    cfg: DetectionConfig
    history_buckets: int = field(init=False)
    warmup_buckets: int = field(init=False)
    level_buckets: int = field(init=False)

    def __post_init__(self) -> None:
        per_day = pd.Timedelta(days=1) / bucket_step(self.cfg.freq)
        self.history_buckets = int(round(self.cfg.stream_history_days * per_day))
        self.warmup_buckets = int(round(self.cfg.stream_warmup_days * per_day))
        self.level_buckets = self.cfg.stream_level_buckets
        minimum = 2 * self.cfg.period + 1
        if self.warmup_buckets < minimum:
            raise ValueError(f"stream warm-up must cover at least {minimum} buckets (two daily cycles)")

    # ----------------------------------------------------------------- scoring
    def _forecast_rule(self, past: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        """Apply the forecast rule to ``past`` (see the module docstring).

        Returns (a) the stabilised errors the rule makes on past buckets t >= lag, each
        forecast from the buckets before t only, and (b) the forecast for the bucket
        right after ``past``.
        """
        # MSTL's combined pattern repeats every week; STL's every day
        lag = self.cfg.weekly_period if uses_mstl(len(past), self.cfg) else self.cfg.period
        past_forecast: dict[str, np.ndarray] = {}
        forecast: dict[str, float] = {}
        for kpi in KPI_COLUMNS:
            values = past[kpi].to_numpy(dtype=float)
            if kpi != "conversion_rate":
                values = np.log1p(values)  # multiplicative seasonality, as in batch
            _trend, seasonal = decompose(values, self.cfg)
            deseasonalised = pd.Series(values - seasonal)
            # level for bucket t = median of the deseasonalised buckets just before t
            levels = deseasonalised.rolling(self.level_buckets, min_periods=1).median().shift(1).to_numpy()
            fitted = levels[lag:] + seasonal[:-lag]
            nxt = float(np.median(deseasonalised.to_numpy()[-self.level_buckets :])) + float(seasonal[-lag])
            if kpi == "conversion_rate":
                past_forecast[f"{kpi}_expected"] = np.clip(fitted, 1e-6, 1 - 1e-6)
                forecast[f"{kpi}_expected"] = float(np.clip(nxt, 1e-6, 1 - 1e-6))
            else:
                past_forecast[f"{kpi}_expected"] = np.maximum(np.expm1(fitted), 0.0)
                forecast[f"{kpi}_expected"] = max(float(np.expm1(nxt)), 0.0)
        tail = past.iloc[lag:]
        errors = stabilised_residuals(tail, pd.DataFrame(past_forecast, index=tail.index)[EXPECTED_COLUMNS])
        return errors, forecast

    def _training_features(self, kpis: pd.DataFrame, i: int) -> np.ndarray:
        """In-sample IF features (as in batch) of the history ending at bucket i.

        We also tried fitting the forest on the rule's past forecast errors (the same kind
        of feature it scores). On the default scenario (21 days, seeds 42 and 1) that gave
        more false kpi_drop windows (3 and 2 -> 5 and 2 at 25 level buckets; 6 and 6 at
        6 level buckets), so the forest stays on the batch features.
        """
        hist = kpis.iloc[max(0, i + 1 - self.history_buckets) : i + 1]
        return if_features(residual_zscores(hist, expected_kpis(hist, self.cfg)), self.cfg.rolling_window)

    def _live_bucket(self, kpis: pd.DataFrame, i: int) -> tuple[dict[str, float], np.ndarray]:
        """Forecast the expected KPIs of bucket i from the buckets before it; return them and i's z-scores."""
        errors, forecast = self._forecast_rule(kpis.iloc[max(0, i - self.history_buckets) : i])
        new = stabilised_residuals(kpis.iloc[i : i + 1], pd.DataFrame(forecast, index=kpis.index[i : i + 1]))
        # scaled by the median and MAD of the rule's past errors, so noise gets |z| ~ 1
        z = np.array([robust_z(new[c].to_numpy(), errors[c].to_numpy())[0] for c in Z_COLUMNS])
        return forecast, z

    def _refit_point(self, i: int) -> int:
        """The latest refit position <= i on the grid warmup + k * stream_refit_every."""
        k = (i - self.warmup_buckets) // self.cfg.stream_refit_every
        return self.warmup_buckets + k * self.cfg.stream_refit_every

    def replay(
        self,
        kpis: pd.DataFrame,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        on_alert: Callable[[StreamAlert], None] | None = None,
    ) -> StreamResult:
        """Score buckets ``start <= bucket < end`` (default: everything after the warm-up)."""
        _check_kpis(kpis)
        t0 = time.perf_counter()
        index = kpis.index
        if len(kpis) <= self.warmup_buckets:
            raise ValueError(
                f"streaming needs more than {self.warmup_buckets} buckets of history "
                f"(stream_warmup_days={self.cfg.stream_warmup_days}), got {len(kpis)}"
            )
        first = self.warmup_buckets
        if start is not None:
            first = max(first, int(index.searchsorted(pd.Timestamp(start), side="left")))
        stop = len(kpis) if end is None else int(index.searchsorted(pd.Timestamp(end), side="left"))
        step = bucket_step(self.cfg.freq)

        tracker = WindowTracker(gap_tolerance=self.cfg.gap_tolerance, min_buckets=self.cfg.min_window_buckets)
        forest: IsolationForest | None = None
        fitted_at = -1
        rows: list[dict[str, object]] = []
        alerts: list[StreamAlert] = []
        n_ids = 0
        span_ids: dict[int, str] = {}  # id(span) -> window id
        # Live z-scores of the few buckets just before `first`, so the rolling features of
        # the first replayed buckets equal those of a replay that started earlier.
        window = self.cfg.rolling_window
        recent_z = [self._live_bucket(kpis, j)[1] for j in range(max(self.warmup_buckets, first - window + 1), first)]

        for i in range(first, max(first, stop)):
            # Refit the forest when bucket i passes a new grid point. It is fitted on the
            # features as they were AT that grid point (not at i), so a replay that starts
            # mid-grid uses the same forest as a full replay would.
            refit_at = self._refit_point(i)
            if forest is None or refit_at != fitted_at:
                forest = fit_isolation_forest(self._training_features(kpis, refit_at), self.cfg)
                fitted_at = refit_at
            expected, z = self._live_bucket(kpis, i)
            recent_z = (recent_z + [z])[-window:]
            rolling = np.mean(recent_z, axis=0)  # trailing mean of the live z-scores

            # Same feature layout as if_features: [z..., rolling mean of z...]
            score = float(forest.decision_function(np.hstack([z, rolling]).reshape(1, -1))[0])
            flagged = score < 0  # IsolationForest.predict == -1 exactly when decision_function < 0
            signals = pd.DataFrame([z + rolling], columns=Z_COLUMNS)  # as direction_signals
            label = str(direction_labels(signals, [flagged])[0])
            row: dict[str, object] = {"bucket": index[i]}
            for col in ("attempts", "successes", "failures", "conversion_rate", "revenue", "avg_latency_ms"):
                if col in kpis.columns:
                    row[col] = kpis[col].iloc[i]
            for n, kpi in enumerate(KPI_COLUMNS):
                row[f"{kpi}_expected"] = expected[f"{kpi}_expected"]
                row[f"{kpi}_resid_z"] = float(z[n])  # Z_COLUMNS follows KPI_COLUMNS order
            row.update(if_score=score, is_anomaly=flagged, label=label)
            rows.append(row)

            confirmed, _closed = tracker.push(len(rows) - 1, label)
            if confirmed is not None:
                n_ids += 1
                span_ids[id(confirmed)] = f"S-{n_ids:03d}"
                alert = self._alert(span_ids[id(confirmed)], confirmed, rows, step)
                alerts.append(alert)
                if on_alert is not None:
                    on_alert(alert)
        tracker.flush()

        scores = pd.DataFrame(rows).set_index("bucket") if rows else _empty_scores(kpis)
        scores.index.name = "bucket"
        windows = [
            make_window(span_ids[id(span)], scores, span, self.cfg.freq, detected_at=_detected_at(scores, span, step))
            for span in tracker.spans
        ]
        runtime = time.perf_counter() - t0
        log.info("stream: %d buckets replayed in %.1fs, %d alerts", len(rows), runtime, len(alerts))
        return StreamResult(alerts=alerts, windows=windows, scores=scores, steps=len(rows), runtime_s=runtime)

    def _alert(self, window_id: str, span: Span, rows: list[dict[str, object]], step: pd.Timedelta) -> StreamAlert:
        frame = pd.DataFrame(rows[span.first : span.confirmed_at + 1]).set_index("bucket")
        start = frame.index[0]
        detected = frame.index[-1] + step
        observed, expected = window_kpis(frame, start, detected)
        return StreamAlert(
            window_id=window_id,
            kpi_type=span.kind,
            start=start.to_pydatetime(),
            detected_at=detected.to_pydatetime(),
            observed=observed,
            expected=expected,
        )


def _detected_at(scores: pd.DataFrame, span: Span, step: pd.Timedelta) -> datetime:
    assert span.confirmed_at is not None
    return (scores.index[span.confirmed_at] + step).to_pydatetime()


def _empty_scores(kpis: pd.DataFrame) -> pd.DataFrame:
    columns = list(kpis.columns)
    for kpi in KPI_COLUMNS:
        columns += [f"{kpi}_expected", f"{kpi}_resid_z"]
    columns += ["if_score", "is_anomaly", "label"]
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="bucket"))


# ------------------------------------------------------------ ground truth


class _IncidentLike(Protocol):
    incident_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class IncidentDetection:
    incident_id: str
    detected: bool
    window_id: str | None
    detected_at: datetime | None
    delay_h: float | None  # detected_at - incident start, hours (None if missed or batch)


def detection_delays(windows: Iterable[AnomalyWindow], incidents: Iterable[_IncidentLike]) -> list[IncidentDetection]:
    """Match kpi_drop windows to incidents by time overlap; delay from the earliest alert."""
    drops = [w for w in windows if w.kpi_type == "kpi_drop"]
    out = []
    for inc in incidents:
        hits = [w for w in drops if w.start < inc.end and w.end > inc.start]
        if not hits:
            out.append(IncidentDetection(inc.incident_id, False, None, None, None))
            continue
        timed = [w for w in hits if w.detected_at is not None]
        first = min(timed, key=lambda w: w.detected_at) if timed else min(hits, key=lambda w: w.start)
        delay = None
        if first.detected_at is not None:
            delay = (pd.Timestamp(first.detected_at) - pd.Timestamp(inc.start)) / pd.Timedelta(hours=1)
        out.append(IncidentDetection(inc.incident_id, True, first.window_id, first.detected_at, delay))
    return out
