"""Load the Kaggle eCommerce behaviour CSVs into hourly KPI tables.

Dataset (``eCommerce behavior data from multi category store``): one row per tracked
event with ``event_time, event_type, product_id, category_id, category_code, brand,
price, user_id, user_session``. It has no country, OS or payment gateway and no explicit
"checkout attempt" row, so this module prepares KPIs only; it does not fabricate the
transaction-level columns that the synthetic generator provides.

Memory. The CSVs are far too large for ``pd.read_csv`` without ``chunksize`` (the Oct and
Nov 2019 files together are about 15 GB). Files are read in chunks with an explicit
``usecols``/``dtype`` list, reduced to per-session flags, and only those (much smaller)
session keys are concatenated. The raw rows are never held for the whole file.

Documented rules
----------------
- **Timestamp.** ``event_time`` (UTC, e.g. ``2019-11-01 00:00:00 UTC``) is parsed as
  UTC, floored to the hour and stored timezone-naive as the project schema requires.
  An unparseable timestamp is an :class:`InvalidEcomDataError`.
- **Events.** ``view``, ``cart``, ``purchase`` and ``remove_from_cart`` are accepted; any
  other label is an :class:`InvalidEcomDataError`. ``remove_from_cart`` is ignored by
  the funnel KPIs below.
- **Duplicates.** Exact duplicate events (same ``event_time, event_type, product_id,
  user_id, user_session, price``) are merged to one *within a chunk*. A duplicate pair
  split across a chunk boundary cannot be detected without holding the whole file; use a
  larger ``chunksize`` if that matters. Session-level flags are merged across all chunks,
  so a session spanning a boundary is still counted once per hour.
- **Sessions.** ``sessions`` is the number of distinct ``user_session`` values in the
  hour. A missing/blank session id cannot be tracked; each such event gets a synthetic
  id so its traffic (and any purchase) is not silently dropped. Session ids are hashed
  to 64-bit integers for memory; a hash collision would merge two sessions (probability
  negligible for this dataset).
- **Purchases.** ``purchases`` is the number of distinct sessions with at least one
  ``purchase`` event (not the number of purchase lines), so ``conversion_rate =
  purchases / sessions`` is a session conversion rate. ``purchase_events`` keeps the
  line count and ``revenue`` sums the price of purchase lines.
- **Cart.** ``carts`` is the number of distinct sessions with at least one ``cart``
  event and ``cart_rate = carts / sessions``.
- **Segments.** The hour x segment table has one row per (hour, attribute, value) for
  three attributes: top-level category (``category_code`` before the first dot),
  brand (global top-N by event count, plus ``other``; a missing brand is ``unknown``),
  and a fixed price band. Missing category/price values become ``unknown``.
- **Missing hours.** The observed range is reindexed to a gap-free hourly grid. Empty
  hours get zero counts and revenue and are flagged ``imputed=True``; their rate columns
  (conversion, cart) are interpolated from neighbours, the same convention as
  :func:`kpi_rca.detection.kpi.aggregate_kpis`. The gaps are returned and logged.

The tables are cached as parquet under ``data/processed/`` (git-ignored) keyed by the
source files and parameters, so repeated runs skip the CSV scan.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

log = logging.getLogger(__name__)

CACHE_VERSION = 1
CACHE_DIR = Path("data/processed")
DEFAULT_CHUNKSIZE = 1_000_000
DEFAULT_TOP_BRANDS = 20
DEFAULT_PRICE_EDGES = (0.0, 25.0, 50.0, 100.0, 200.0, 500.0, np.inf)
DEFAULT_PRICE_LABELS = ("0-25", "25-50", "50-100", "100-200", "200-500", "500+")

REQUIRED_COLUMNS = (
    "event_time",
    "event_type",
    "product_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
)
EVENT_TYPES = frozenset({"view", "cart", "purchase", "remove_from_cart"})
DEDUPE_KEY = ("event_time", "event_type", "product_id", "user_id", "user_session", "price")
SEGMENT_ATTRIBUTES = ("category", "brand", "price_band")
HOURLY_COLUMNS = (
    "sessions",
    "purchases",
    "purchase_events",
    "carts",
    "conversion_rate",
    "cart_rate",
    "revenue",
    "imputed",
)
SEGMENT_COLUMNS = (
    "hour",
    "attribute",
    "value",
    "sessions",
    "purchases",
    "purchase_events",
    "carts",
    "conversion_rate",
    "cart_rate",
    "revenue",
    "imputed",
)
_DTYPES = {
    "event_time": "string",
    "event_type": "string",
    "product_id": "string",
    "category_code": "string",
    "brand": "string",
    "price": "float64",
    "user_id": "string",
    "user_session": "string",
}


class EcomDataError(ValueError):
    """Base class for eCommerce dataset problems (subclass of ValueError for callers)."""


class InvalidEcomDataError(EcomDataError):
    """The CSV or KPI frame is malformed: missing columns, bad timestamps or values."""


class InsufficientEcomDataError(EcomDataError):
    """Fewer hourly buckets than the two seasonal periods STL needs."""


@dataclass
class EcomKPIDataset:
    """Hourly KPIs and the hour x segment table, plus what was imputed and read."""

    hourly: pd.DataFrame
    segments: pd.DataFrame
    gaps: list[dict[str, object]]
    top_brands: list[str]
    sources: list[str] = field(default_factory=list)
    cache_paths: dict[str, Path] = field(default_factory=dict)


# --------------------------------------------------------------------- helpers


def _resolve_paths(csv_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(csv_paths, (str, Path)):
        path = Path(csv_paths)
        if path.is_dir():
            candidates = sorted(path.glob("*.csv"))
            paths = [p for p in candidates if _looks_like_ecom_csv(p)]
        else:
            paths = [path]
    else:
        paths = [Path(p) for p in csv_paths]
    if not paths:
        raise InvalidEcomDataError("no eCommerce CSV files were given or found")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise InvalidEcomDataError(f"eCommerce CSV file(s) not found: {missing}")
    return paths


def _looks_like_ecom_csv(path: Path) -> bool:
    try:
        header = path.open("r", encoding="utf-8", errors="replace").readline()
    except OSError:
        return False
    return all(column in header for column in ("event_time", "event_type", "user_session"))


def _normalise_text(series: pd.Series) -> pd.Series:
    """Strings, stripped; blank/null become ``unknown``."""
    out = series.astype("string").str.strip()
    return out.mask(out.isna() | out.eq(""), "unknown")


def _normalise_brand(series: pd.Series) -> pd.Series:
    return _normalise_text(series)


def _normalise_category(series: pd.Series) -> pd.Series:
    text = _normalise_text(series)
    return text.str.split(".").str[0].fillna("unknown")


def _price_band(price: pd.Series, edges: Sequence[float], labels: Sequence[str]) -> pd.Series:
    band = pd.cut(price, bins=list(edges), labels=list(labels), right=False, include_lowest=True)
    return band.astype("string").fillna("unknown")


def _gap_runs(index: pd.DatetimeIndex, step: pd.Timedelta) -> list[dict[str, object]]:
    if len(index) == 0:
        return []
    gaps: list[dict[str, object]] = []
    start = previous = index[0]
    for timestamp in index[1:]:
        if timestamp - previous == step:
            previous = timestamp
            continue
        gaps.append(
            {
                "start": start.isoformat(),
                "end": (previous + step).isoformat(),
                "hours": int((previous - start) / step) + 1,
            }
        )
        start = previous = timestamp
    gaps.append(
        {
            "start": start.isoformat(),
            "end": (previous + step).isoformat(),
            "hours": int((previous - start) / step) + 1,
        }
    )
    return gaps


def _rate(series: pd.Series, denominator: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float)
    counts = denominator.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.divide(values, counts, out=np.zeros_like(values), where=counts > 0)
    return pd.Series(rate, index=series.index)


def _fill_rates(frame: pd.DataFrame, columns: Sequence[str], missing: pd.Series) -> pd.DataFrame:
    """Reindexed frame: zero counts stay zero, rate columns interpolate and are clipped."""
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype(float)
        values[missing.to_numpy(dtype=bool)] = np.nan  # only missing hours are interpolated
        values = values.interpolate(method="linear", limit_direction="both").clip(0.0, 1.0)
        frame[column] = values.fillna(0.0)
    frame["imputed"] = missing
    return frame


# ---------------------------------------------------------------------- loader


class KaggleEcomLoader:
    """Chunked CSV -> hourly KPI tables, with a parquet cache under ``data/processed``.

    ``top_brands`` may be an int (compute the global top-N by event count) or an explicit
    list of brand names (skips the extra brand-count pass). ``price_edges``/``price_labels``
    define the price bands. ``chunksize`` trades memory for the chance of a duplicate pair
    crossing a chunk boundary (see the module docstring).
    """

    def __init__(
        self,
        csv_paths: str | Path | Iterable[str | Path],
        *,
        cache_dir: str | Path | None = CACHE_DIR,
        chunksize: int = DEFAULT_CHUNKSIZE,
        top_brands: int | Sequence[str] = DEFAULT_TOP_BRANDS,
        price_edges: Sequence[float] = DEFAULT_PRICE_EDGES,
        price_labels: Sequence[str] = DEFAULT_PRICE_LABELS,
        freq: str = "1h",
    ) -> None:
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1")
        if len(price_edges) != len(price_labels) + 1:
            raise ValueError("price_edges must have one more entry than price_labels")
        self.paths = _resolve_paths(csv_paths)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.chunksize = int(chunksize)
        self.top_brands = top_brands
        self.price_edges = tuple(float(v) for v in price_edges)
        self.price_labels = tuple(str(v) for v in price_labels)
        self.freq = freq
        self._step = pd.Timedelta(to_offset(freq).nanos, unit="ns")

    # ------------------------------------------------------------- public API
    def load(self, *, use_cache: bool = True, force: bool = False) -> EcomKPIDataset:
        """Return the hourly and segment tables, reading the parquet cache when valid."""
        key = self._cache_key()
        paths = self._cache_paths(key)
        if use_cache and not force and paths and all(p.is_file() for p in paths.values()):
            dataset = self._read_cache(paths)
            log.info("loaded eCommerce KPIs from cache %s", paths["hourly"].parent)
            return dataset

        hourly, segments, gaps, top_brands = self._aggregate()
        dataset = EcomKPIDataset(
            hourly=hourly,
            segments=segments,
            gaps=gaps,
            top_brands=top_brands,
            sources=[str(p) for p in self.paths],
            cache_paths=paths or {},
        )
        if paths:
            self._write_cache(dataset, paths)
        return dataset

    # ------------------------------------------------------------- cache
    def _cache_key(self) -> str:
        sources = [
            {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in self.paths
        ]
        top = list(self.top_brands) if isinstance(self.top_brands, (list, tuple, set)) else int(self.top_brands)
        payload = {
            "version": CACHE_VERSION,
            "sources": sources,
            "params": {
                "chunksize": self.chunksize,
                "top_brands": top,
                "price_edges": list(self.price_edges),
                "price_labels": list(self.price_labels),
                "freq": self.freq,
            },
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _cache_paths(self, key: str) -> dict[str, Path] | None:
        if self.cache_dir is None:
            return None
        return {
            "hourly": self.cache_dir / f"ecom_hourly_{key}.parquet",
            "segments": self.cache_dir / f"ecom_segments_{key}.parquet",
            "metadata": self.cache_dir / f"ecom_metadata_{key}.json",
        }

    def _read_cache(self, paths: dict[str, Path]) -> EcomKPIDataset:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        hourly = pd.read_parquet(paths["hourly"])
        hourly.index = pd.DatetimeIndex(hourly.index, name="hour", freq=metadata.get("freq"))
        segments = pd.read_parquet(paths["segments"])
        return EcomKPIDataset(
            hourly=hourly,
            segments=segments,
            gaps=list(metadata.get("gaps", [])),
            top_brands=list(metadata.get("top_brands", [])),
            sources=list(metadata.get("sources", [])),
            cache_paths=paths,
        )

    def _write_cache(self, dataset: EcomKPIDataset, paths: dict[str, Path]) -> None:
        paths["hourly"].parent.mkdir(parents=True, exist_ok=True)
        dataset.hourly.to_parquet(paths["hourly"])
        dataset.segments.to_parquet(paths["segments"], index=False)
        metadata = {
            "version": CACHE_VERSION,
            "gaps": dataset.gaps,
            "top_brands": dataset.top_brands,
            "sources": dataset.sources,
            "freq": self.freq,
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- reading
    def _iter_chunks(self, usecols: Sequence[str], dtypes: dict[str, str] | None = None) -> Iterable[pd.DataFrame]:
        dtype = {k: v for k, v in (dtypes or _DTYPES).items() if k in usecols}
        for path in self.paths:
            try:
                reader = pd.read_csv(
                    path,
                    usecols=list(usecols),
                    dtype=dtype,
                    chunksize=self.chunksize,
                    low_memory=False,
                )
                for chunk in reader:
                    yield chunk
            except (ValueError, KeyError, pd.errors.ParserError) as exc:
                raise InvalidEcomDataError(f"cannot read eCommerce CSV {path}: {exc}") from exc

    def _resolve_top_brands(self) -> list[str]:
        if isinstance(self.top_brands, (list, tuple, set)):
            return [str(value) for value in self.top_brands]
        n = int(self.top_brands)
        if n <= 0:
            return []
        counts: Counter[str] = Counter()
        for chunk in self._iter_chunks(["brand"]):
            brands = _normalise_brand(chunk["brand"])
            counts.update(brands.value_counts().to_dict())
        counts.pop("unknown", None)
        return [name for name, _ in counts.most_common(n)]

    # --------------------------------------------------------- aggregation
    def _aggregate(self) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], list[str]]:
        top_brands = self._resolve_top_brands()
        top_set = set(top_brands)
        total_parts: list[pd.DataFrame] = []
        segment_parts: list[pd.DataFrame] = []
        row_offset = 0
        duplicate_rows = 0
        missing_sessions = 0

        for chunk in self._iter_chunks(REQUIRED_COLUMNS):
            missing_columns = [column for column in REQUIRED_COLUMNS if column not in chunk.columns]
            if missing_columns:
                raise InvalidEcomDataError(
                    f"eCommerce CSV is missing required columns {missing_columns}; the data is not the expected dataset"
                )
            prepared, segment_rows, stats = self._prepare_chunk(chunk, row_offset, top_set)
            duplicate_rows += stats["duplicate_rows"]
            missing_sessions += stats["missing_sessions"]
            row_offset += stats["input_rows"]

            total_parts.append(prepared)
            segment_parts.append(segment_rows)

        if not total_parts:
            raise InvalidEcomDataError("no events were read from the eCommerce CSV file(s)")
        log.info(
            "read %d rows, dropped %d duplicate events, synthesised %d missing session ids",
            row_offset,
            duplicate_rows,
            missing_sessions,
        )

        total = pd.concat(total_parts, ignore_index=True)
        if total.empty:
            raise InvalidEcomDataError("no usable events remain after timestamp/type/price validation")
        total = total.groupby(["hour", "_session"], observed=True, as_index=False).agg(
            had_purchase=("had_purchase", "max"),
            had_cart=("had_cart", "max"),
            purchase_events=("purchase_events", "sum"),
            revenue=("revenue", "sum"),
        )
        hourly = total.groupby("hour", observed=True, as_index=False).agg(
            sessions=("_session", "size"),
            purchases=("had_purchase", "sum"),
            purchase_events=("purchase_events", "sum"),
            carts=("had_cart", "sum"),
            revenue=("revenue", "sum"),
        )

        segments = pd.concat(segment_parts, ignore_index=True)
        segments = segments.groupby(["hour", "attribute", "value", "_session"], observed=True, as_index=False).agg(
            had_purchase=("had_purchase", "max"),
            had_cart=("had_cart", "max"),
            purchase_events=("purchase_events", "sum"),
            revenue=("revenue", "sum"),
        )
        segments = segments.groupby(["hour", "attribute", "value"], observed=True, as_index=False).agg(
            sessions=("_session", "size"),
            purchases=("had_purchase", "sum"),
            purchase_events=("purchase_events", "sum"),
            carts=("had_cart", "sum"),
            revenue=("revenue", "sum"),
        )

        hourly = self._finish_hourly(hourly)
        gaps = _gap_runs(hourly.index[hourly["imputed"].to_numpy(dtype=bool)], self._step)
        if gaps:
            lost_hours = sum(int(gap["hours"]) for gap in gaps)
            log.warning("filled %d missing hour(s) in %d gap(s); see dataset.gaps", lost_hours, len(gaps))
        segments = self._finish_segments(segments, hourly.index)
        log.info(
            "aggregated %d hourly buckets and %d segment rows (top brands: %s)",
            len(hourly),
            len(segments),
            ", ".join(top_brands) if top_brands else "(none)",
        )
        return hourly, segments, gaps, top_brands

    def _prepare_chunk(
        self, chunk: pd.DataFrame, row_offset: int, top_set: set[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
        input_rows = len(chunk)
        timestamps = pd.to_datetime(chunk["event_time"], utc=True, errors="coerce")
        bad_timestamps = timestamps.isna()
        if bad_timestamps.any():
            example = chunk.loc[bad_timestamps, "event_time"].iloc[0]
            raise InvalidEcomDataError(
                f"{int(bad_timestamps.sum())} event_time value(s) are not valid UTC timestamps, e.g. {example!r}"
            )
        hours = timestamps.dt.tz_convert(None).dt.floor(self.freq)

        event_type = chunk["event_type"].astype("string").str.strip().str.lower()
        bad_type = event_type.isna() | ~event_type.isin(EVENT_TYPES)
        if bad_type.any():
            example = chunk.loc[bad_type.fillna(True), "event_type"].iloc[0]
            raise InvalidEcomDataError(
                f"{int(bad_type.sum())} event_type value(s) are not in {sorted(EVENT_TYPES)}, e.g. {example!r}"
            )

        price = pd.to_numeric(chunk["price"], errors="coerce").astype(float)
        if np.isinf(price.to_numpy()).any():
            raise InvalidEcomDataError("price has infinite values")
        negative = price < 0
        if negative.any():
            raise InvalidEcomDataError(f"price has {int(negative.sum())} negative value(s)")
        purchase = event_type.eq("purchase")
        missing_purchase_price = purchase & price.isna()
        if missing_purchase_price.any():
            raise InvalidEcomDataError(
                f"{int(missing_purchase_price.sum())} purchase event(s) have a missing or non-numeric price"
            )

        duplicates = int(chunk.duplicated(subset=list(DEDUPE_KEY)).sum())
        if duplicates:
            chunk = chunk.drop_duplicates(subset=list(DEDUPE_KEY))
            timestamps = timestamps.loc[chunk.index]
            hours = hours.loc[chunk.index]
            event_type = event_type.loc[chunk.index]
            price = price.loc[chunk.index]
            purchase = purchase.loc[chunk.index]

        session = chunk["user_session"].astype("string").str.strip()
        missing_session = session.isna() | session.eq("")
        missing_count = int(missing_session.sum())
        if missing_count:
            synthetic = pd.Series(
                np.arange(row_offset, row_offset + len(chunk), dtype=np.int64), index=chunk.index
            ).astype("string")
            session = session.where(~missing_session, "__missing_session_" + synthetic)
        session_id = pd.util.hash_pandas_object(session, index=False).to_numpy(dtype=np.uint64)

        category = _normalise_category(chunk["category_code"])
        brand = _normalise_brand(chunk["brand"])
        brand = brand.where(brand.isin(top_set) | brand.eq("unknown"), "other")
        band = _price_band(price, self.price_edges, self.price_labels)

        purchase_values = purchase.fillna(False).to_numpy(dtype=bool)
        cart_values = event_type.eq("cart").fillna(False).to_numpy(dtype=bool)
        revenue = np.where(purchase_values, price.fillna(0.0).to_numpy(dtype=float), 0.0)

        total = pd.DataFrame(
            {
                "hour": hours.to_numpy(),
                "_session": session_id,
                "had_purchase": purchase_values,
                "had_cart": cart_values,
                "purchase_events": purchase_values.astype(np.int64),
                "revenue": revenue,
            }
        )
        total = total.groupby(["hour", "_session"], observed=True, as_index=False).agg(
            had_purchase=("had_purchase", "max"),
            had_cart=("had_cart", "max"),
            purchase_events=("purchase_events", "sum"),
            revenue=("revenue", "sum"),
        )

        pieces = []
        for attribute, values in (("category", category), ("brand", brand), ("price_band", band)):
            pieces.append(
                pd.DataFrame(
                    {
                        "hour": hours.to_numpy(),
                        "attribute": attribute,
                        "value": values.to_numpy(dtype=object),
                        "_session": session_id,
                        "had_purchase": purchase_values,
                        "had_cart": cart_values,
                        "purchase_events": purchase_values.astype(np.int64),
                        "revenue": revenue,
                    }
                )
            )
        segments = pd.concat(pieces, ignore_index=True)
        segments = segments.groupby(["hour", "attribute", "value", "_session"], observed=True, as_index=False).agg(
            had_purchase=("had_purchase", "max"),
            had_cart=("had_cart", "max"),
            purchase_events=("purchase_events", "sum"),
            revenue=("revenue", "sum"),
        )

        stats = {"input_rows": input_rows, "duplicate_rows": duplicates, "missing_sessions": missing_count}
        return total.reset_index(drop=True), segments.reset_index(drop=True), stats

    def _finish_hourly(self, hourly: pd.DataFrame) -> pd.DataFrame:
        hourly = hourly.set_index("hour").sort_index()
        hourly.index.name = "hour"
        observed_index = hourly.index
        full_index = pd.date_range(observed_index.min(), observed_index.max(), freq=self.freq, name="hour")
        missing = pd.Series(~full_index.isin(observed_index), index=full_index, name="imputed")
        frame = hourly.reindex(full_index)
        for column in ("sessions", "purchases", "purchase_events", "carts", "revenue"):
            frame[column] = frame[column].fillna(0)
        frame["conversion_rate"] = _rate(frame["purchases"], frame["sessions"])
        frame["cart_rate"] = _rate(frame["carts"], frame["sessions"])
        frame = _fill_rates(frame, ("conversion_rate", "cart_rate"), missing)
        frame["sessions"] = frame["sessions"].astype("int64")
        frame["purchases"] = frame["purchases"].astype("int64")
        frame["purchase_events"] = frame["purchase_events"].astype("int64")
        frame["carts"] = frame["carts"].astype("int64")
        frame["revenue"] = frame["revenue"].astype(float)
        return frame[list(HOURLY_COLUMNS)]

    def _finish_segments(self, segments: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
        frames = []
        for (attribute, value), group in segments.groupby(["attribute", "value"], sort=True, observed=True):
            group = group.set_index("hour").sort_index()
            observed_index = group.index
            missing = pd.Series(~full_index.isin(observed_index), index=full_index, name="imputed")
            frame = group.reindex(full_index)
            for column in ("sessions", "purchases", "purchase_events", "carts", "revenue"):
                frame[column] = frame[column].fillna(0)
            frame["conversion_rate"] = _rate(frame["purchases"], frame["sessions"])
            frame["cart_rate"] = _rate(frame["carts"], frame["sessions"])
            frame = _fill_rates(frame, ("conversion_rate", "cart_rate"), missing)
            frame["attribute"] = attribute
            frame["value"] = value
            frame.index.name = "hour"
            frames.append(frame.reset_index())
        if not frames:
            return pd.DataFrame(columns=list(SEGMENT_COLUMNS))
        result = pd.concat(frames, ignore_index=True)
        for column in ("sessions", "purchases", "purchase_events", "carts"):
            result[column] = result[column].astype("int64")
        result["revenue"] = result["revenue"].astype(float)
        return result[list(SEGMENT_COLUMNS)]


# --------------------------------------------------------------- public helpers


def load_kaggle_ecom_hourly(
    csv_paths: str | Path | Iterable[str | Path],
    *,
    cache_dir: str | Path | None = CACHE_DIR,
    chunksize: int = DEFAULT_CHUNKSIZE,
    top_brands: int | Sequence[str] = DEFAULT_TOP_BRANDS,
    freq: str = "1h",
    period: int = 24,
    validate: bool = True,
) -> pd.DataFrame:
    """Hourly KPI frame for eCommerce CSVs (convenience wrapper around the loader).

    Set ``validate=False`` to skip the regular-grid/seasonal-length checks (tests use
    short fixtures); by default the frame must cover at least ``2 * period + 1`` hours.
    """
    dataset = KaggleEcomLoader(
        csv_paths,
        cache_dir=cache_dir,
        chunksize=chunksize,
        top_brands=top_brands,
        freq=freq,
    ).load()
    if validate:
        validate_ecom_kpis(dataset.hourly, period=period, freq=freq)
    return dataset.hourly


def validate_ecom_kpis(
    hourly: pd.DataFrame,
    *,
    period: int = 24,
    freq: str = "1h",
) -> pd.DataFrame:
    """Validate an eCommerce hourly KPI frame and return a cleaned copy.

    Checks the timezone-naive UTC hourly index is sorted, duplicate-free and a regular
    ``freq`` grid; that counts and revenue are finite and non-negative; that
    ``purchases <= sessions`` and the rates lie in ``[0, 1]``; and that there are at
    least ``2 * period + 1`` buckets, the minimum STL needs for two seasonal cycles.
    """
    if not isinstance(hourly, pd.DataFrame):
        raise InvalidEcomDataError(f"hourly KPI table must be a DataFrame, got {type(hourly).__name__}")
    missing_columns = [column for column in ("sessions", "purchases", "conversion_rate", "cart_rate", "revenue") if column not in hourly.columns]
    if missing_columns:
        raise InvalidEcomDataError(f"hourly KPI table is missing columns {missing_columns}")

    frame = hourly.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "hour" in frame.columns:
            frame = frame.set_index("hour")
        else:
            raise InvalidEcomDataError("hourly KPI table needs a DatetimeIndex named 'hour'")
    index = frame.index
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    frame.index = index
    frame.index.name = "hour"
    if frame.empty:
        raise InsufficientEcomDataError("hourly KPI table is empty; at least two seasonal periods are needed")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index(kind="stable")
    if frame.index.has_duplicates:
        raise InvalidEcomDataError("hourly KPI table has duplicate hours")

    numeric_columns = ("sessions", "purchases", "purchase_events", "carts", "conversion_rate", "cart_rate", "revenue")
    check_columns = [column for column in numeric_columns if column in frame.columns]
    values = frame[check_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [column for column in check_columns if not np.isfinite(frame[column].to_numpy(dtype=float)).all()]
        raise InvalidEcomDataError(f"hourly KPI table has NaN or infinite values in {bad}")
    negative = [column for column in check_columns if (frame[column].to_numpy(dtype=float) < 0).any()]
    if negative:
        raise InvalidEcomDataError(f"hourly KPI table has negative values in {negative}")
    if "purchases" in frame.columns and (frame["purchases"].to_numpy(dtype=float) > frame["sessions"].to_numpy(dtype=float)).any():
        raise InvalidEcomDataError("hourly KPI table has purchases greater than sessions")
    for column in ("conversion_rate", "cart_rate"):
        if column in frame.columns:
            values = frame[column].to_numpy(dtype=float)
            outside = (values < 0) | (values > 1)
            if outside.any():
                raise InvalidEcomDataError(f"hourly KPI table has {int(outside.sum())} {column} values outside [0, 1]")

    expected_index = pd.date_range(frame.index.min(), frame.index.max(), freq=freq)
    if len(expected_index) != len(frame.index) or not expected_index.equals(frame.index):
        raise InvalidEcomDataError(
            f"hourly KPI table is not a regular {freq!r} grid; missing hours must be filled and flagged 'imputed'"
        )
    minimum = 2 * period + 1
    if len(frame) < minimum:
        raise InsufficientEcomDataError(
            f"need at least {minimum} hourly buckets (two full seasonal periods of {period} buckets) "
            f"to score, got {len(frame)}; load more history or lower detection.period"
        )
    return frame


def build_kpi_frame_for_pipeline(
    hourly: pd.DataFrame,
    *,
    period: int = 24,
    freq: str = "1h",
    validate: bool = True,
) -> pd.DataFrame:
    """Map an eCommerce hourly table onto the detection/ pipeline KPI column names.

    ``sessions -> attempts``, ``purchases -> successes``, ``failures = attempts -
    successes``; ``conversion_rate``, ``revenue`` and ``imputed`` keep their names and
    ``cart_rate`` is carried for context. The Kaggle data has no latency, so
    ``avg_latency_ms`` is intentionally absent (detection does not use it and inventing
    a value would be fabricated data). The index is renamed ``bucket`` to match
    :func:`kpi_rca.detection.kpi.aggregate_kpis`.
    """
    frame = validate_ecom_kpis(hourly, period=period, freq=freq) if validate else hourly.copy()
    if "sessions" not in frame.columns or "purchases" not in frame.columns:
        raise InvalidEcomDataError("hourly KPI table needs 'sessions' and 'purchases' columns")
    attempts = frame["sessions"].astype("int64")
    successes = frame["purchases"].astype("int64")
    out = pd.DataFrame(
        {
            "attempts": attempts,
            "successes": successes,
            "failures": attempts - successes,
            "conversion_rate": frame["conversion_rate"].astype(float),
            "revenue": frame["revenue"].astype(float),
            "cart_rate": frame["cart_rate"].astype(float),
            "imputed": frame.get("imputed", pd.Series(False, index=frame.index)).astype(bool),
        },
        index=frame.index,
    )
    out.index.name = "bucket"
    return out
