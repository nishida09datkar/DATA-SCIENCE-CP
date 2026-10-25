"""Aggregate checkout attempts into hourly (or ``freq``) KPI time series.

Documented preprocessing rules (they define what detection sees):

- **Duplicate timestamps.** Event rows whose floor(bucket) falls in the same bucket are
  merged by an aggregation rule, not deduplicated: counts (attempts, successes) and
  revenue are summed, latency is averaged, and the conversion rate of the merged bucket
  is recomputed from the merged successes over merged attempts. The same rule is applied
  to a raw KPI frame with repeated buckets by :func:`preprocess_kpis` (detect) - counts
  and revenue summed, latency averaged, conversion recomputed - so two paths agree.
- **Missing values.** Empty buckets (no attempts in [start, end), all-NaN columns) have
  zero counts and revenue; the rate-like columns (conversion_rate, avg_latency_ms) are
  linearly interpolated across the gap (nearest values carried at the edges), and the
  ``imputed`` column marks every bucket that received any interpolated value. Detection
  only imputes rate-like columns: a zero count with an interpolated rate carries no
  weight in the conversion z-score (its attempts of 0 make the binomial sigma tiny).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

KPI_FRAME_COLUMNS = [
    "attempts",
    "successes",
    "failures",
    "conversion_rate",
    "revenue",
    "avg_latency_ms",
    "imputed",
]
COUNT_COLUMNS = ("attempts", "successes", "revenue")  # duplicates merge by summation
MEAN_COLUMNS = ("conversion_rate", "avg_latency_ms")  # ratio/mean columns merge by mean


def aggregate_kpis(
    tx: pd.DataFrame,
    freq: str = "1h",
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Bucket attempts by ``timestamp`` floored to ``freq``.

    Returns a frame indexed by a gap-free DatetimeIndex named ``bucket`` (bucket start)
    with attempts, successes, failures, conversion_rate (successes/attempts), revenue
    (sum of successful amounts), avg_latency_ms and ``imputed``. The index spans the
    data, or ``[start, end)`` when given. Empty buckets get zero counts and revenue;
    their conversion rate and latency are interpolated from neighbouring buckets so the
    decomposition sees a continuous series, and their ``imputed`` value is True (see the
    module docstring for the rules).
    """
    if tx.empty and (start is None or end is None):
        raise ValueError("cannot aggregate an empty transaction frame without an explicit start and end")
    bucket = tx["timestamp"].dt.floor(freq)
    success = tx["success"].to_numpy(dtype=np.int64)
    frame = pd.DataFrame(
        {
            "bucket": bucket.to_numpy(),
            "attempts": np.ones(len(tx), dtype=np.int64),
            "successes": success,
            "revenue": tx["amount"].to_numpy(dtype=float) * success,
            "latency": tx["latency_ms"].to_numpy(dtype=float),
        }
    )
    grouped = frame.groupby("bucket", sort=True).agg(
        attempts=("attempts", "sum"),
        successes=("successes", "sum"),
        revenue=("revenue", "sum"),
        avg_latency_ms=("latency", "mean"),
    )

    first = pd.Timestamp(start).floor(freq) if start is not None else grouped.index.min()
    if end is not None:
        index = pd.date_range(first, pd.Timestamp(end), freq=freq, inclusive="left", name="bucket")
    else:
        index = pd.date_range(first, grouped.index.max(), freq=freq, name="bucket")
    index = index.astype("datetime64[ns]")
    if len(index) == 0:
        raise ValueError(f"no {freq} buckets between {first} and {end}")
    if start is not None or end is not None:
        outside = (grouped.index < index[0]) | (grouped.index > index[-1])
        if outside.any():
            raise ValueError(f"{int(grouped.loc[outside, 'attempts'].sum())} attempts fall outside [{first}, {end})")
    grouped = grouped.reindex(index)

    out = pd.DataFrame(index=index)
    out["attempts"] = grouped["attempts"].fillna(0).astype(np.int64)
    out["successes"] = grouped["successes"].fillna(0).astype(np.int64)
    out["failures"] = out["attempts"] - out["successes"]
    # conversion is recomputed on the merged bucket; empty buckets interpolate below
    with np.errstate(divide="ignore", invalid="ignore"):
        conversion = out["successes"] / out["attempts"].where(out["attempts"] > 0)
    conversion_filled, conv_imputed = _fill_gaps(conversion)
    latency_filled, lat_imputed = _fill_gaps(grouped["avg_latency_ms"].astype(float))
    out["conversion_rate"] = conversion_filled
    out["revenue"] = grouped["revenue"].fillna(0.0).astype(float)
    out["avg_latency_ms"] = latency_filled
    out["imputed"] = conv_imputed | lat_imputed
    return out[KPI_FRAME_COLUMNS]


def _fill_gaps(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Linearly interpolate missing values; report which points were interpolated."""
    missing = series.isna()
    if not missing.any():
        return series.astype(float), pd.Series(False, index=series.index)
    if missing.all():
        # nothing to interpolate from: fall back to 0 and mark everything imputed
        return series.fillna(0.0).astype(float), missing
    filled = series.interpolate(method="linear", limit_direction="both").astype(float)
    return filled, missing


def preprocess_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw KPI frame for detection; the batch pipeline's documented rules.

    Takes any frame indexed by timestamps holding at least attempt counts, revenue and a
    conversion rate, and returns a frame ``detect_anomalies`` can decompose:

    1. the index becomes a sorted, ascending DatetimeIndex;
    2. duplicate buckets are merged by the aggregation rule in the module docstring
       (counts and revenue summed, rate/latency averaged, conversion recomputed);
    3. missing rate-like values are linearly interpolated and marked in ``imputed``;
    4. remaining NaN/inf values and impossible values (negative counts or revenue,
       conversion outside [0, 1]) raise :class:`kpi_rca.detection.detector.InvalidKPIError`.
    """
    from kpi_rca.detection.detector import InvalidKPIError  # avoids a circular import at module load

    _require_kpi_columns(frame)
    frame = frame.sort_index(kind="stable")
    if frame.index.has_duplicates:
        n = len(frame)
        sums = frame[[c for c in COUNT_COLUMNS if c in frame.columns]].groupby(level=0, sort=True).sum()
        means = frame[[c for c in MEAN_COLUMNS if c in frame.columns]].groupby(level=0, sort=True).mean()
        merged = sums.join(means)
        if "successes" in merged.columns and "attempts" in merged.columns:
            merged["conversion_rate"] = merged["successes"] / merged["attempts"].where(merged["attempts"] > 0)
        if "attempts" in merged.columns and "successes" in merged.columns:
            merged["failures"] = merged["attempts"] - merged["successes"]
        frame = merged
        if "imputed" in frame.columns:
            frame["imputed"] = frame["imputed"].astype(bool)
        missing_after = n - len(frame)
    if "imputed" not in frame.columns:
        frame["imputed"] = False
    for column in ("conversion_rate", "avg_latency_ms"):
        if column not in frame.columns:
            continue
        filled, imputed = _fill_gaps(frame[column].astype(float))
        frame[column] = filled
        frame["imputed"] = frame["imputed"].astype(bool) | imputed
    _reject_impossible_values(frame, InvalidKPIError)
    return frame


def _require_kpi_columns(frame: pd.DataFrame) -> None:
    from kpi_rca.detection.detector import InvalidKPIError

    needed = [c for c in ("conversion_rate", "revenue", "attempts") if c not in frame.columns]
    if needed:
        raise InvalidKPIError(f"KPI frame is missing columns {needed} (aggregate transactions with aggregate_kpis)")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise InvalidKPIError("KPI frame needs a DatetimeIndex (bucket-start timestamps)")


def _reject_impossible_values(frame: pd.DataFrame, error: type[Exception]) -> None:
    check_cols = [c for c in KPI_FRAME_COLUMNS if c in frame.columns and c != "imputed"]
    values = frame[check_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [c for c in check_cols if not np.isfinite(frame[c].to_numpy(dtype=float)).all()]
        details = "; ".join(f"{c} has non-finite values at {list(frame.index[~np.isfinite(frame[c].to_numpy(dtype=float))][:3])}" for c in bad)  # fmt: skip
        raise error(f"KPI frame has NaN or infinite values after preprocessing: {details}")
    negative = {
        c: int((frame[c].to_numpy(dtype=float) < 0).sum())
        for c in ("attempts", "successes", "failures", "revenue", "avg_latency_ms")
        if c in frame.columns and (frame[c].to_numpy(dtype=float) < 0).sum() > 0
    }
    if negative:
        raise error(f"KPI frame has negative values where none are possible: {negative} (counts and revenue >= 0)")
    conversion = frame["conversion_rate"].to_numpy(dtype=float)
    bad_conversion = (conversion < 0) | (conversion > 1)
    if bad_conversion.any():
        raise error(
            f"KPI frame has {int(bad_conversion.sum())} conversion_rate values outside [0, 1], "
            f"e.g. {float(conversion[bad_conversion][0]):.4g} at {frame.index[bad_conversion][0]}"
        )
