"""Batch anomaly detection on KPI series: STL/MSTL residuals + Isolation Forest.

Per KPI (conversion_rate, revenue, attempts):

1. Decompose with robust STL - statsmodels' STL(robust=True), i.e. STL with robustness
   weights, not the separate RobustSTL algorithm of Wen et al. (2019) - using the daily period, a wide seasonal smoother
   (``stl_seasonal`` daily cycles) and a trend about one day wide. With at least
   ``mstl_min_weeks`` weeks of buckets, MSTL with daily *and* weekly periods is used
   instead (default: six weeks; the measurements behind that choice are next to
   ``uses_mstl``). Until then the day-level weekly effect is left to the STL trend, which
   tracks it; robust weights keep multi-hour incidents out of the trend. attempts and
   revenue are decomposed on a log1p scale because their seasonality is multiplicative.
   expected = trend + seasonal (back-transformed).
2. Variance-stabilised residuals - conversion: residual / binomial sigma given the
   observed attempts; attempts and revenue: residual / sqrt(expected attempts), the
   Poisson scale - so quiet night hours and busy evening hours are comparable. A robust
   z = (x - median) / (1.4826 * MAD) puts every KPI on the same scale.
3. Isolation Forest (contamination="auto", fixed random_state) on the z-scores and their
   short trailing rolling means. No manual thresholds on KPI values.
4. Direction-aware labels, read from each KPI's z + trailing rolling mean (the same
   signals the forest saw): volume_surge when attempts are up by more sigmas than
   conversion or revenue is down; kpi_drop when conversion or revenue is down otherwise.
   Flagged buckets with neither direction (e.g. conversion up) keep the label "normal".
5. Consecutive flagged buckets with the same label, allowing ``gap_tolerance`` other
   buckets in between, form a window. A window needs ``min_window_buckets`` flagged
   buckets: Isolation Forest with contamination="auto" flags a few isolated noise
   buckets in any long series, and this debounce removes them.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import MSTL, STL

from kpi_rca.config import DetectionConfig
from kpi_rca.detection.kpi import preprocess_kpis
from kpi_rca.schemas import KPI_COLUMNS, AnomalyWindow, DetectionResult

log = logging.getLogger(__name__)

MAD_SCALE = 1.4826
FLOAT_NOISE_REL = 1e-9  # differences below this relative size are log/exp roundtrip noise
LABELS = ("normal", "kpi_drop", "volume_surge")
Z_COLUMNS = [f"{k}_resid_z" for k in KPI_COLUMNS]
EXPECTED_COLUMNS = [f"{k}_expected" for k in KPI_COLUMNS]
WEEKLY_SEASONAL_WINDOW = 101  # MSTL weekly smoother length in weeks (odd, > any history)


# --------------------------------------------------------------- typed errors


class InvalidKPIError(ValueError):
    """A KPI frame cannot be scored: missing columns, unsorted index, bad or duplicate values.

    Subclass of ValueError, so callers that catch ValueError keep working.
    """


class InsufficientDataError(ValueError):
    """Not enough observations for the decomposition (STL needs two full seasonal periods)."""


class ConstantSeriesError(ValueError):
    """Series is constant and cannot be decomposed for anomaly detection."""


class NaNSeriesError(ValueError):
    """Series contains NaN or infinite values."""


# ------------------------------------------------------------- decomposition


# STL or MSTL? MSTL(24, 168) adds a weekly component with one value per hour of the week
# (168 phases). Each phase is estimated from one observation per week, so three weeks of
# history give only 3 observations per phase: the robust fit becomes a median of three,
# which follows the noise (the residual MAD shrinks and every z-score is inflated), and a
# non-robust fit absorbs a third of each incident into the "weekly pattern" instead.
#
# Measured on the default scenario (seeds 42 and 1): false |z| > 4 on normal hours,
# conversion / attempts, robust STL(24, seasonal=21, trend=25) vs the MSTL used below
# (windows=(21, 101), seasonal_deg=0, robust):
#
#     history   STL(24)          MSTL(24, 168)
#     21 days   5/0 and 0/1      39/41 and 13/20
#     28 days   0/0 and 1/1       9/5  and  7/12
#     42 days   2/2 and 6/7       5/5  and  8/8
#
# statsmodels' default MSTL windows are worse still: at 21 days the MAD collapses to ~0
# (INC-001 z = -9633) and INC-002 disappears into the weekly component (z = -0.1); at
# 42 days they still give 80-105 false conversion alarms. Both incidents stay clearly
# visible with STL (z -7 to -14). The generator's weekly effect is a day-level
# multiplier, which the one-day STL trend tracks. Hence the default switch at six weeks
# (>= 6 observations per phase), where MSTL is on a par with STL and can pay off for real
# data with weekday-specific hourly shapes.
def uses_mstl(n_buckets: int, cfg: DetectionConfig) -> bool:
    """MSTL(daily, weekly) once ``mstl_min_weeks`` weeks of buckets exist, else STL(daily)."""
    return n_buckets >= cfg.mstl_min_weeks * cfg.weekly_period


def _check_series(values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise InsufficientDataError("series is empty")
    if not np.all(np.isfinite(arr)):
        raise NaNSeriesError("series contains NaN or infinite values")
    if np.allclose(arr, arr[0]):
        raise ConstantSeriesError("series is constant")




def decompose(values, cfg):
    """(trend, seasonal) of a series: robust STL, or MSTL with enough weeks."""
    _check_series(values)
    n = values.shape[0]
    if n < 2 * cfg.period + 1:
        raise InsufficientDataError(f"need at least {2 * cfg.period + 1} buckets (two full seasonal periods of {cfg.period} buckets) for STL, got {n}; collect more history or lower detection.period")
    if uses_mstl(n, cfg):
        from statsmodels.tsa.seasonal import MSTL
        result = MSTL(values, periods=(cfg.period, cfg.weekly_period), windows=(cfg.stl_seasonal, WEEKLY_SEASONAL_WINDOW), stl_kwargs={"robust": True, "seasonal_deg": 0}).fit()
        seasonal = np.asarray(result.seasonal, dtype=float)
        if seasonal.ndim == 2:
            seasonal = seasonal.sum(axis=1)
    else:
        from statsmodels.tsa.seasonal import STL
        result = STL(values, period=cfg.period, seasonal=cfg.stl_seasonal, trend=stl_trend_window(cfg), robust=True).fit()
        seasonal = np.asarray(result.seasonal, dtype=float)
    return np.asarray(result.trend, dtype=float), seasonal

def _trend_plus_seasonal(values: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    trend, seasonal = decompose(values, cfg)
    return trend + seasonal


def stl_trend_window(cfg: DetectionConfig) -> int:
    """Trend smoother length: ``stl_trend`` or the smallest odd number above one period."""
    if cfg.stl_trend is not None:
        return cfg.stl_trend
    return cfg.period + 1 if cfg.period % 2 == 0 else cfg.period + 2


def decomposition_name(n_buckets: int, cfg: DetectionConfig) -> str:
    if uses_mstl(n_buckets, cfg):
        return f"MSTL({cfg.period},{cfg.weekly_period})"
    return f"STL({cfg.period})"


def auto_period_from_freq(freq: str | pd.tseries.offsets.BaseOffset) -> int:
    try:
        if isinstance(freq, str):
            off = pd.tseries.frequencies.to_offset(freq)
        else:
            off = freq
        if off is None:
            return 24
        name = getattr(off, "name", str(off)).lower()
        if "h" in name or "hour" in name:
            return 24
        if "d" in name or "day" in name:
            return 7
        if "w" in name or "week" in name:
            return 168
    except Exception:
        pass
    return 24


def resolve_period(cfg: DetectionConfig, index: pd.DatetimeIndex | None = None, allow_weekly: bool = False) -> int:
    period = int(getattr(cfg, "period", 24))
    if index is not None and hasattr(index, "freq") and index.freq is not None:
        inferred = auto_period_from_freq(index.freq)
        if period == 24 and inferred == 7:
            period = 7
        elif period == 24 and inferred == 168:
            period = 168
        elif inferred == 168 and allow_weekly:
            period = 168
    if allow_weekly and period == 24:
        period = 168
    return period


def expected_kpis(kpis: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    """Expected value of every KPI in KPI_COLUMNS from its trend + seasonal components."""
    out = {}
    conversion = kpis["conversion_rate"].to_numpy(dtype=float)
    try:
        conv_exp = _trend_plus_seasonal(conversion, cfg)
    except ConstantSeriesError:
        conv_exp = conversion.copy()
    out["conversion_rate_expected"] = np.clip(conv_exp, 1e-6, 1 - 1e-6)
    for kpi in ("revenue", "attempts"):
        try:
            fit = _trend_plus_seasonal(np.log1p(kpis[kpi].to_numpy(dtype=float)), cfg)
            out[f"{kpi}_expected"] = np.maximum(np.expm1(fit), 0.0)
        except ConstantSeriesError:
            out[f"{kpi}_expected"] = kpis[kpi].to_numpy(dtype=float).copy()
    return pd.DataFrame(out, index=kpis.index)[EXPECTED_COLUMNS]


def _real_diff(value: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """value - expected, with differences below FLOAT_NOISE_REL*max(|value|) zeroed out.

    The back-transformed expected values (expm1 of an STL fit on log1p) carry ~1e-15
    floating-point error. On an exactly constant series that error is ALL the residual:
    a residual MAD around 1e-10 would rescale pure roundtrip noise to |z| around 19 and
    the forest would flag its artifacts. Real deviations are many orders of magnitude
    larger, so the threshold only acts where there is no signal at all.
    """
    diff = np.asarray(value, dtype=float) - np.asarray(expected, dtype=float)
    tiny = np.abs(diff) <= FLOAT_NOISE_REL * np.maximum(np.abs(value), np.abs(expected))
    diff[tiny] = 0.0
    return diff


def stabilised_residuals(kpis: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Residuals divided by their sampling scale (before the robust rescaling)."""
    attempts = kpis["attempts"].to_numpy(dtype=float)
    p = expected["conversion_rate_expected"].to_numpy(dtype=float)
    sigma = np.sqrt(p * (1.0 - p) / np.maximum(attempts, 1.0))
    conversion = np.where(attempts > 0, (kpis["conversion_rate"].to_numpy(dtype=float) - p) / sigma, 0.0)
    scale = np.sqrt(np.maximum(expected["attempts_expected"].to_numpy(dtype=float), 1.0))
    raw = {
        "conversion_rate_resid_z": conversion,
        "revenue_resid_z": _real_diff(kpis["revenue"].to_numpy(dtype=float), expected["revenue_expected"].to_numpy())
        / scale,
        "attempts_resid_z": _real_diff(attempts, expected["attempts_expected"].to_numpy()) / scale,
    }
    return pd.DataFrame(raw, index=kpis.index)[Z_COLUMNS]


def robust_residual_zscore(residual: np.ndarray, reference: np.ndarray | None = None, mask_valid: np.ndarray | None = None) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    ref = residual if reference is None else np.asarray(reference, dtype=float)
    if mask_valid is not None:
        mv = np.asarray(mask_valid, dtype=bool)
        if mv.any():
            ref_use = ref[mv]
        else:
            ref_use = ref
    else:
        ref_use = ref
    med = float(np.nanmedian(ref_use))
    mad = float(np.nanmedian(np.abs(ref_use - med))) * MAD_SCALE
    if not np.isfinite(mad) or mad < 1e-12:
        sd = float(np.nanstd(ref_use))
        mad = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    return (residual - med) / mad




def robust_z(values, reference=None):
    return robust_residual_zscore(values, reference=reference)

def residual_zscores(kpis: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    raw = stabilised_residuals(kpis, expected)
    return raw.apply(lambda col: pd.Series(robust_z(col.to_numpy()), index=col.index))


# --------------------------------------------------------- isolation forest


def if_features(z: pd.DataFrame, rolling_window: int) -> np.ndarray:
    """IF inputs, derived from the STL residual via its robust z-scores (documented layout).

    Three blocks, stacked in this order and always built through this function so batch
    and streaming never diverge:

    1. the robust z-scores themselves (residual in 1.4826*MAD units);
    2. their trailing rolling mean (causal; keeps a strong anomaly visible for up to
       ``rolling_window - 1`` buckets after it ends);
    3. their first difference (how fast the residual is moving; separates the start of a
       drop, where the residual plunges, from a stable tail inside it).
    """
    zz = z[Z_COLUMNS]
    rolling = zz.rolling(rolling_window, min_periods=1).mean()
    diff = zz.diff().fillna(0.0)
    return np.hstack([zz.to_numpy(dtype=float), rolling.to_numpy(dtype=float), diff.to_numpy(dtype=float)])


def fit_isolation_forest(features: np.ndarray, cfg: DetectionConfig) -> IsolationForest:
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # feature-name warnings on plain arrays
        model.fit(features)
    return model


def max_abs_z(z: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Largest absolute robust z over the KPIs of each bucket (NaN counts as 0)."""
    values = np.asarray(z, dtype=float)
    values = values if values.ndim == 2 else values.reshape(1, -1)
    absz = np.abs(np.nan_to_num(values, nan=0.0))
    return absz.max(axis=1)


def combine_signals(flagged_if: np.ndarray, z: pd.DataFrame, cfg: DetectionConfig) -> np.ndarray:
    """The documented IF+STL flag rule (data-driven thresholds, no KPI-value cut-offs).

    A bucket is flagged when the Isolation Forest isolates it AND the largest |residual
    z| reaches ``min_gate_z`` (the two signals agree), OR when the largest |residual z|
    itself reaches ``extreme_z`` even where the forest did not isolate it (one signal is
    extreme). Both thresholds live on the robust z scale (1.4826*MAD units), which is
    derived from the series' own median residual scale, so no constant sits on KPI
    values; the gate removes the low-z noise that contamination="auto" flags in any long
    series, and the extreme clause keeps a huge residual visible if the forest misses it.
    """
    gate = np.asarray(flagged_if, dtype=bool) & (max_abs_z(z[Z_COLUMNS]) >= cfg.min_gate_z)
    extreme = max_abs_z(z[Z_COLUMNS]) >= cfg.extreme_z
    return gate | extreme


def point_reason(
    z_row: np.ndarray,
    signal_row: np.ndarray,
    if_score: float,
    kpi_row: np.ndarray,
    expected_row: np.ndarray,
    kind: str = "",
) -> str:
    """Human-readable reason for one flagged bucket (empty string for normal buckets).

    Names the residual that defines the bucket's direction - attempts for a volume_surge
    (the direction_labels rule), conversion/revenue for a kpi_drop, otherwise the largest
    |signal| - plus its MAD-scaled z, the IF score and the signed gap to the seasonal
    expectation, e.g. "residual -7.2 MAD in conversion_rate, IF score 0.31,
    conversion_rate 12% below seasonal expectation".
    """
    if not np.isfinite(signal_row).all() or len(KPI_COLUMNS) != len(signal_row):
        return ""
    if kind == "volume_surge":
        best = KPI_COLUMNS.index("attempts")
    elif kind == "kpi_drop":
        drops = [KPI_COLUMNS.index(k) for k in ("conversion_rate", "revenue")]
        best = max(drops, key=lambda j: -signal_row[j])
    else:
        best = int(np.argmax(np.abs(signal_row)))
    if signal_row[best] == 0 and z_row[best] == 0:
        return ""
    kpi = KPI_COLUMNS[best]
    gap = kpi_row[best] - expected_row[best]
    if expected_row[best] != 0 and np.isfinite(expected_row[best]):
        deviation = f"{kpi} {gap / expected_row[best] * 100.0:+.0f}% vs seasonal expectation"
    elif gap < 0:
        deviation = f"{kpi} below seasonal expectation by {gap:+.4g}"
    else:
        deviation = f"{kpi} above seasonal expectation by {gap:+.4g}"
    return f"residual {z_row[best]:+.1f} MAD in {kpi}, IF score {if_score:.2f}, {deviation}"


def direction_signals(z: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Per-KPI direction of what the forest saw: z-score + its trailing rolling mean.

    The rolling-mean features keep a strong anomaly visible for up to ``rolling_window - 1``
    buckets after it ends, and the forest flags those buckets too. Reading the direction
    from the same two features keeps them with the anomaly they belong to; the current z
    alone would give them the sign of whatever noise the bucket has.
    """
    zz = z[Z_COLUMNS]
    return zz + zz.rolling(rolling_window, min_periods=1).mean()


def direction_labels(signals: pd.DataFrame, flagged: np.ndarray) -> np.ndarray:
    """Label flagged buckets by which deviation dominates (all signals are in sigma units).

    shortfall = how far conversion or revenue is below expectation (the larger of the two).
    volume_surge: attempts are up by more sigmas than the shortfall (more traffic, possibly
    with a little mix-driven conversion dip); kpi_drop: otherwise any shortfall (a
    conversion drop while traffic is also up is still a drop); else normal.
    """
    dc = signals["conversion_rate_resid_z"].to_numpy()
    dr = signals["revenue_resid_z"].to_numpy()
    da = signals["attempts_resid_z"].to_numpy()
    shortfall = np.maximum(np.maximum(-dc, -dr), 0.0)
    surge = da > shortfall
    drop = ~surge & (shortfall > 0)
    labels = np.where(surge, "volume_surge", np.where(drop, "kpi_drop", "normal"))
    return np.where(np.asarray(flagged, dtype=bool), labels, "normal").astype(object)


# ------------------------------------------------------------------ windows


@dataclass
class Span:
    kind: str  # kpi_drop | volume_surge
    first: int  # index of the first flagged bucket
    last: int  # index of the last flagged bucket
    n_flagged: int = 1
    confirmed_at: int | None = None  # index of the bucket that completed the debounce


@dataclass
class WindowTracker:
    """Online grouping of labelled buckets into windows (shared by batch and streaming)."""

    gap_tolerance: int
    min_buckets: int
    open: Span | None = None
    spans: list[Span] = field(default_factory=list)

    def push(self, i: int, label: str) -> tuple[Span | None, Span | None]:
        """Feed bucket ``i``; returns (span confirmed by this bucket, span closed before it)."""
        closed = None
        confirmed = None
        if label in ("kpi_drop", "volume_surge"):
            cur = self.open
            if cur is not None and cur.kind == label and i - cur.last - 1 <= self.gap_tolerance:
                cur.last = i
                cur.n_flagged += 1
            else:
                closed = self._close()
                cur = self.open = Span(kind=label, first=i, last=i)
            if cur.confirmed_at is None and cur.n_flagged >= self.min_buckets:
                cur.confirmed_at = i
                confirmed = cur
        elif self.open is not None and i - self.open.last > self.gap_tolerance:
            closed = self._close()
        return confirmed, closed

    def flush(self) -> Span | None:
        return self._close()

    def _close(self) -> Span | None:
        cur, self.open = self.open, None
        if cur is not None and cur.confirmed_at is not None:
            self.spans.append(cur)
            return cur
        return None


def group_windows(labels: np.ndarray, gap_tolerance: int, min_buckets: int) -> list[Span]:
    tracker = WindowTracker(gap_tolerance=gap_tolerance, min_buckets=min_buckets)
    for i, label in enumerate(labels):
        tracker.push(i, label)
    tracker.flush()
    return tracker.spans


def window_kpis(scores: pd.DataFrame, start: datetime, end: datetime) -> tuple[dict[str, float], dict[str, float]]:
    """Observed and expected KPI totals over buckets ``start <= bucket < end``."""
    rows = scores.loc[(scores.index >= start) & (scores.index < end)]
    attempts = float(rows["attempts"].sum())
    observed = {
        "conversion_rate": float(rows["successes"].sum() / attempts) if attempts > 0 else float("nan"),
        "revenue": float(rows["revenue"].sum()),
        "attempts": attempts,
    }
    weights = rows["attempts"].to_numpy(dtype=float)
    conv_exp = rows["conversion_rate_expected"].to_numpy(dtype=float)
    expected = {
        "conversion_rate": float(np.average(conv_exp, weights=weights)) if weights.sum() > 0 else float(conv_exp.mean()),
        "revenue": float(rows["revenue_expected"].sum()),
        "attempts": float(rows["attempts_expected"].sum()),
    }
    return observed, expected


def make_window(
    window_id: str,
    scores: pd.DataFrame,
    span: Span,
    freq: str,
    detected_at: datetime | None = None,
) -> AnomalyWindow:
    step = pd.tseries.frequencies.to_offset(freq)
    start = scores.index[span.first]
    end = scores.index[span.last] + step
    observed, expected = window_kpis(scores, start, end)
    if_score_min = float(scores["if_score"].iloc[span.first : span.last + 1].min())
    peak_score = float(scores["anomaly_score"].iloc[span.first : span.last + 1].max())
    peak_pos = int(scores["anomaly_score"].iloc[span.first : span.last + 1].to_numpy().argmax()) + span.first
    reason = ""
    if "reason" in scores.columns:
        reason = str(scores["reason"].iloc[peak_pos])
    drop_pct = None
    primary = "attempts" if span.kind == "volume_surge" else "conversion_rate"
    exp = expected[primary]
    if np.isfinite(exp) and exp != 0 and np.isfinite(observed[primary]):
        drop_pct = (exp - observed[primary]) / exp  # negative when the KPI moved the other way
    return AnomalyWindow(
        window_id=window_id,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        kpi_type=span.kind,
        n_buckets=span.last - span.first + 1,
        if_score_min=if_score_min,
        observed=observed,
        expected=expected,
        detected_at=detected_at,
        duration_hours=(end - start).total_seconds() / 3600.0,
        peak_anomaly_score=max(peak_score, 0.0) if np.isfinite(peak_score) else 0.0,
        drop_pct=drop_pct if drop_pct is not None and np.isfinite(drop_pct) else None,
        reason=reason,
    )


# -------------------------------------------------------------------- batch


def detect_anomalies(kpis: pd.DataFrame, cfg: DetectionConfig) -> DetectionResult:
    """Score every bucket and group flagged buckets into AnomalyWindows (batch mode).

    A raw KPI frame is cleaned first by :func:`kpi_rca.detection.kpi.preprocess_kpis`
    (sorted index, duplicate buckets merged, missing rate values interpolated and
    flagged); then the decomposition + Isolation Forest + the documented combination
    rule in :func:`combine_signals` decide the flags.
    """
    kpis = preprocess_kpis(kpis)
    _check_kpis(kpis)
    expected = expected_kpis(kpis, cfg)
    z = residual_zscores(kpis, expected)
    features = if_features(z, cfg.rolling_window)
    forest = fit_isolation_forest(features, cfg)
    if_score = forest.decision_function(features)
    flagged = combine_signals(forest.predict(features) == -1, z, cfg)

    signals = direction_signals(z, cfg.rolling_window)
    scores = kpis.copy()
    for kpi in KPI_COLUMNS:
        scores[f"{kpi}_expected"] = expected[f"{kpi}_expected"]
        scores[f"{kpi}_resid_z"] = z[f"{kpi}_resid_z"]
    scores["if_score"] = if_score.astype(float)
    scores["anomaly_score"] = (-np.asarray(if_score, dtype=float)).astype(float)  # higher = more anomalous
    scores["is_anomaly"] = flagged
    scores["label"] = direction_labels(signals, flagged)
    # schema compatibility
    scores["direction"] = scores["label"]
    scores.index.name = "bucket"
    # add timestamp alias
    scores["timestamp"] = scores.index
    scores["reason"] = [
        point_reason(
            z[Z_COLUMNS].to_numpy(dtype=float)[i],
            signals[Z_COLUMNS].to_numpy(dtype=float)[i],
            float(if_score[i]),
            kpis[list(KPI_COLUMNS)].to_numpy(dtype=float)[i],
            expected[EXPECTED_COLUMNS].to_numpy(dtype=float)[i],
            kind=str(scores["label"].iloc[i]),
        )
        if flagged[i]
        else ""
        for i in range(len(scores))
    ]  # flagged buckets explain themselves (empty for normal buckets)
    # add kpi_value/expected_value aliases (primary KPI or conversion_rate as default)
    scores["kpi_value"] = scores["conversion_rate"] if "conversion_rate" in scores.columns else scores.get(KPI_COLUMNS[0])
    scores["expected_value"] = scores["conversion_rate_expected"] if "conversion_rate_expected" in scores.columns else scores.get(f"{KPI_COLUMNS[0]}_expected")
    if "metadata" not in scores.columns:
        scores["metadata"] = [{} for _ in range(len(scores))]

    spans = group_windows(scores["label"].to_numpy(), cfg.gap_tolerance, cfg.min_window_buckets)
    windows = [make_window(f"W-{n:03d}", scores, span, cfg.freq) for n, span in enumerate(spans, start=1)]
    log.info(
        "detection: %d buckets (%s), %d flagged, %d windows (%d kpi_drop, %d volume_surge)",
        len(scores),
        decomposition_name(len(scores), cfg),
        int(flagged.sum()),
        len(windows),
        sum(w.kpi_type == "kpi_drop" for w in windows),
        sum(w.kpi_type == "volume_surge" for w in windows),
    )
    return DetectionResult(scores=scores, windows=windows)


def _check_kpis(kpis: pd.DataFrame) -> None:
    missing = [c for c in ("attempts", "successes", "conversion_rate", "revenue") if c not in kpis.columns]
    if missing:
        raise InvalidKPIError(f"KPI frame is missing columns {missing}")
    if not isinstance(kpis.index, pd.DatetimeIndex) or not kpis.index.is_monotonic_increasing:
        raise InvalidKPIError("KPI frame needs a sorted DatetimeIndex (use aggregate_kpis)")
    if kpis.index.has_duplicates:
        raise InvalidKPIError(
            "KPI frame has duplicate bucket timestamps; batch detection merges them "
            "(preprocess_kpis: counts/revenue summed, latency averaged, conversion recomputed), "
            "streaming needs unique buckets - use aggregate_kpis"
        )
    values = kpis[["conversion_rate", "revenue", "attempts"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [c for c in ("conversion_rate", "revenue", "attempts") if not np.isfinite(kpis[c].to_numpy(dtype=float)).all()]
        raise InvalidKPIError(
            f"KPI frame has NaN or infinite values in {bad}; use aggregate_kpis or preprocess_kpis, "
            "which interpolate and mark imputed buckets"
        )
    negative = [
        c for c in ("attempts", "successes", "failures", "revenue") if c in kpis.columns and (kpis[c].to_numpy(dtype=float) < 0).any()
    ]
    if negative:
        raise InvalidKPIError(f"KPI frame has negative values in {negative} (counts and revenue must be >= 0)")
    conversion = kpis["conversion_rate"].to_numpy(dtype=float)
    outside = (conversion < 0) | (conversion > 1)
    if outside.any():
        raise InvalidKPIError(
            f"KPI frame has {int(outside.sum())} conversion_rate values outside [0, 1], "
            f"e.g. {float(conversion[outside][0]):.4g} at {kpis.index[outside][0]}"
        )



def robust_stl_decompose(values, cfg=None, period=None, allow_weekly=False, index=None):
    from kpi_rca.config import DetectionConfig as _DC
    if cfg is None:
        cfg = _DC()
    if period is not None:
        cfg = cfg.model_copy(update={"period": int(period)})
    _check_series(values)
    arr = np.asarray(values, dtype=float).ravel()
    p = resolve_period(cfg, index=index, allow_weekly=allow_weekly)
    cfg2 = cfg.model_copy(update={"period": int(p)}) if p != cfg.period else cfg
    n = arr.shape[0]
    if n < 2 * cfg2.period + 1:
        raise InsufficientDataError(f"need at least {2 * cfg2.period + 1} buckets (two full seasonal periods of {cfg2.period} buckets) for STL, got {n}; collect more history or lower detection.period")
    if uses_mstl(n, cfg2):
        result = MSTL(values, periods=(cfg2.period, cfg2.weekly_period), windows=(cfg2.stl_seasonal, WEEKLY_SEASONAL_WINDOW), stl_kwargs={"robust": True, "seasonal_deg": 0}).fit()
        seasonal = np.asarray(result.seasonal, dtype=float)
        if seasonal.ndim == 2:
            seasonal = seasonal.sum(axis=1)
    else:
        result = STL(values, period=cfg2.period, seasonal=cfg2.stl_seasonal, trend=stl_trend_window(cfg2), robust=True).fit()
        seasonal = np.asarray(result.seasonal, dtype=float)
    trend = np.asarray(result.trend, dtype=float)
    seasonal = np.asarray(seasonal, dtype=float)
    expected_value = trend + seasonal
    residual = arr - expected_value
    return {"trend": trend, "seasonal": seasonal, "resid": residual, "residual": residual, "expected_value": expected_value, "expected": expected_value}
