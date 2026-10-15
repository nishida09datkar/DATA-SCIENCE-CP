"""Aggregate checkout attempts into hourly (or ``freq``) KPI time series."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

KPI_FRAME_COLUMNS = ["attempts", "successes", "failures", "conversion_rate", "revenue", "avg_latency_ms"]


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
    (sum of successful amounts) and avg_latency_ms. The index spans the data, or
    ``[start, end)`` when given. Empty buckets get zero counts and revenue; their
    conversion rate and latency are interpolated from neighbouring buckets so the
    decomposition sees a continuous series (their attempts of 0 carry no weight in the
    conversion z-score).
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
    with np.errstate(divide="ignore", invalid="ignore"):
        conversion = out["successes"] / out["attempts"].where(out["attempts"] > 0)
    out["conversion_rate"] = _fill_gaps(conversion)
    out["revenue"] = grouped["revenue"].fillna(0.0).astype(float)
    out["avg_latency_ms"] = _fill_gaps(grouped["avg_latency_ms"].astype(float))
    return out[KPI_FRAME_COLUMNS]


def _fill_gaps(series: pd.Series) -> pd.Series:
    if series.notna().all():
        return series.astype(float)
    if series.isna().all():
        return series.fillna(0.0).astype(float)
    return series.interpolate(method="linear", limit_direction="both").astype(float)
