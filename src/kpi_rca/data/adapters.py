"""Map a real dataset onto the pipeline's transaction and ticket schema.

REES46 (Open CDP "eCommerce behavior data from multi category store", Kaggle
mkechinov/ecommerce-behavior-data-from-multi-category-store) is a raw event log
(view / cart / purchase). One user_session with a cart or purchase event becomes
one checkout attempt:

    transaction_id  <- user_session
    timestamp       <- event_time of its first cart or purchase event (UTC, made naive)
    success         <- 1 if the session has a purchase event, else 0
    amount          <- sum of its purchase prices, else the price of its first cart event
    error_code      <- "NONE" on success, else NO_PURCHASE (REES46 records no error codes)
    latency_ms      <- 0.0 (not recorded)
    category        <- top level of category_code ("unknown" when empty)
    brand           <- brand, the top 20 of the month kept, the rest "other"
    price_band      <- price quartile of the month: p1_low, p2, p3, p4_high

So a "failure" here is a cart without a purchase (abandonment), not a payment error.
The generator's country / os / device_type / payment_gateway / app_version have no
source in REES46 and are not invented: config/rees46.yaml sets dataset: external
and rca.attributes to REES46_ATTRIBUTES.

Tickets: REES46 holds no text at all (no tickets, reviews or complaints).
no_tickets() is an empty ticket frame, so retrieval is skipped and the memo cites no
customers. Nothing is fabricated.

Incidents: REES46 has no labelled incidents. inject_incident() fails a share of the
successful attempts of one segment in one time range, on a copy; the original frame
is never changed. Injected rows get the same NO_PURCHASE code as real failures, so
the data itself does not reveal them; the caller keeps the returned mask as the label.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kpi_rca.data.io import REQUIRED_TICKET_COLUMNS

REES46_ATTRIBUTES = ["category", "brand", "price_band"]
REES46_COLUMNS = ["event_time", "event_type", "category_code", "brand", "price", "user_session"]
NO_PURCHASE = "NO_PURCHASE"
TOP_BRANDS = 20
PRICE_BANDS = ["p1_low", "p2", "p3", "p4_high"]


def load_rees46_events(raw: str | Path, cache: str | Path | None = None) -> pd.DataFrame:
    """Cart and purchase events of one monthly REES46 file (.csv or .csv.gz), time-sorted.

    With ``cache`` (a .parquet path), a cached copy is read when present and written otherwise.
    """
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    raw = Path(raw)
    if not raw.is_file():
        raise ValueError(f"REES46 file not found: {raw}")
    parts = []
    reader = pd.read_csv(raw, usecols=REES46_COLUMNS, dtype={"event_type": "category", "brand": "str"}, chunksize=4_000_000)
    for chunk in reader:
        parts.append(chunk[chunk["event_type"].isin(["cart", "purchase"])])
    events = pd.concat(parts, ignore_index=True)
    events["event_time"] = pd.to_datetime(events["event_time"].str.removesuffix(" UTC"))
    events["event_type"] = events["event_type"].astype(str)
    events = events.dropna(subset=["user_session"]).sort_values("event_time", kind="stable", ignore_index=True)
    if cache is not None:
        events.to_parquet(cache, index=False)
    return events


def rees46_transactions(events: pd.DataFrame, top_brands: int = TOP_BRANDS) -> pd.DataFrame:
    """One checkout attempt per session, in the transaction schema (see the module docstring)."""
    missing = [c for c in REES46_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"REES46 events are missing columns {missing}")
    if events.empty:
        raise ValueError("REES46 events are empty")
    first = events.groupby("user_session", sort=False).first()
    purchases = events[events["event_type"] == "purchase"].groupby("user_session")["price"].sum()
    success = first.index.isin(purchases.index)
    amount = np.where(success, purchases.reindex(first.index).to_numpy(), first["price"].to_numpy())

    category = first["category_code"].fillna("").str.split(".").str[0].replace("", "unknown")
    brand = first["brand"].fillna("unknown")
    brand = brand.where(brand.isin(brand.value_counts().index[:top_brands]), "other")
    try:
        price_band = pd.qcut(first["price"], 4, labels=PRICE_BANDS).astype(str)
    except ValueError:  # many tied prices give duplicate quartile edges: split the ties by order
        price_band = pd.qcut(first["price"].rank(method="first"), 4, labels=PRICE_BANDS).astype(str)

    tx = pd.DataFrame(
        {
            "transaction_id": first.index.astype(str),
            "timestamp": first["event_time"].to_numpy(),
            "category": category.to_numpy(),
            "brand": brand.to_numpy(),
            "price_band": price_band.to_numpy(),
            "amount": amount.astype(float),
            "success": success.astype(np.int64),
            "error_code": np.where(success, "NONE", NO_PURCHASE),
            "latency_ms": 0.0,
        }
    )
    return tx.sort_values("timestamp", kind="stable", ignore_index=True)


def no_tickets() -> pd.DataFrame:
    """An empty ticket frame: the dataset has no text, and none is made up."""
    frame = pd.DataFrame({c: pd.Series(dtype="str") for c in REQUIRED_TICKET_COLUMNS})
    frame["created_at"] = pd.Series(dtype="datetime64[ns]")
    return frame


def candidate_segments(
    tx: pd.DataFrame, attributes: Sequence[str], lo: float = 0.03, hi: float = 0.30
) -> list[dict[str, str]]:
    """Segments of 1 or 2 attributes holding lo..hi of the attempts (no 'other'/'unknown' values)."""
    out = []
    n = len(tx)
    pairs = [[a, b] for i, a in enumerate(attributes) for b in attributes[i + 1:]]
    for attrs in [[a] for a in attributes] + pairs:
        for values, count in tx.groupby(attrs, observed=True).size().items():
            values = values if isinstance(values, tuple) else (values,)
            if lo <= count / n <= hi and not {"other", "unknown"} & set(map(str, values)):
                out.append(dict(zip(attrs, map(str, values))))
    return out


def inject_incident(
    tx: pd.DataFrame,
    segment: dict[str, str],
    start: datetime,
    end: datetime,
    failure_rate: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """A copy of ``tx`` in which each successful attempt of ``segment`` in [start, end)
    fails with probability ``failure_rate``; returns (copy, mask of the flipped rows)."""
    unknown = [a for a in segment if a not in tx.columns]
    if unknown:
        raise ValueError(f"segment names columns that are not in the data: {unknown}")
    if not start < end:
        raise ValueError(f"incident start {start} must be before its end {end}")
    if not 0 < failure_rate <= 1:
        raise ValueError(f"failure_rate must be in (0, 1], got {failure_rate}")
    out = tx.copy()
    mask = (out["timestamp"] >= start) & (out["timestamp"] < end) & (out["success"] == 1)
    for attribute, value in segment.items():
        mask &= out[attribute].astype(str) == value
    hit = mask.to_numpy() & (rng.random(len(out)) < failure_rate)
    out.loc[hit, "success"] = 0
    out.loc[hit, "error_code"] = NO_PURCHASE
    return out, hit
