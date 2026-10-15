"""Scenario persistence and input validation at the system boundary.

A data directory holds ``transactions.parquet``, ``tickets.parquet`` and, for
generated data, ``ground_truth.json``. ``validate_transactions`` and
``validate_tickets`` check everything the pipeline relies on and raise a ValueError
that names the column and the problem; they return a cleaned, time-sorted copy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from kpi_rca.config import SEGMENT_ATTRIBUTES
from kpi_rca.data.generator import TICKET_COLUMNS, TRANSACTION_COLUMNS, GroundTruth, Scenario

log = logging.getLogger(__name__)

TRANSACTIONS_FILE = "transactions.parquet"
TICKETS_FILE = "tickets.parquet"
GROUND_TRUTH_FILE = "ground_truth.json"

REQUIRED_TICKET_COLUMNS = TICKET_COLUMNS[:6]  # gt_incident_id is optional (generated data only)
_OPTIONAL_TICKET_TEXT = ("customer_country", "device_os")


def save_scenario(scenario: Scenario, out_dir: str | Path) -> dict[str, Path]:
    """Write the scenario to ``out_dir``; returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "transactions": out / TRANSACTIONS_FILE,
        "tickets": out / TICKETS_FILE,
    }
    scenario.transactions.to_parquet(paths["transactions"], index=False)
    scenario.tickets.to_parquet(paths["tickets"], index=False)
    if scenario.ground_truth is not None:
        paths["ground_truth"] = out / GROUND_TRUTH_FILE
        paths["ground_truth"].write_text(scenario.ground_truth.model_dump_json(indent=2), encoding="utf-8")
    return paths


def load_scenario(data_dir: str | Path, *, validate: bool = True) -> Scenario:
    """Read a data directory written by :func:`save_scenario` (ground truth optional)."""
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"data directory not found: {root}")
    missing = [name for name in (TRANSACTIONS_FILE, TICKETS_FILE) if not (root / name).is_file()]
    if missing:
        raise ValueError(f"data directory {root} is missing {missing}; run `kpi-rca generate` first")
    tx = pd.read_parquet(root / TRANSACTIONS_FILE)
    tickets = pd.read_parquet(root / TICKETS_FILE)
    truth = None
    gt_path = root / GROUND_TRUTH_FILE
    if gt_path.is_file():
        try:
            truth = GroundTruth.model_validate_json(gt_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{gt_path} is not a valid ground-truth file: {exc}") from None
    if validate:
        tx = validate_transactions(tx)
        tickets = validate_tickets(tickets)
    return Scenario(transactions=tx, tickets=tickets, ground_truth=truth)


# ---------------------------------------------------------------- validation


def _require_frame(df: object, what: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{what} must be a pandas DataFrame, got {type(df).__name__}")
    return df


def _require_columns(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing required columns {missing}; got {list(df.columns)}")


def _naive_utc_times(series: pd.Series, column: str, what: str) -> pd.Series:
    if not is_datetime64_any_dtype(series):
        raise ValueError(f"{what}.{column} must be a datetime column, got dtype {series.dtype}")
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    if series.isna().any():
        raise ValueError(f"{what}.{column} has {int(series.isna().sum())} missing timestamps")
    return series.astype("datetime64[ns]")


def _label_column(series: pd.Series, column: str, what: str, *, allow_missing: bool) -> pd.Series:
    """Validate a text-label column (string, object or categorical dtype), vectorised.

    Returns a categorical column for categorical input and a ``str`` column otherwise.
    Blank labels become missing when ``allow_missing``; otherwise they are an error.
    """
    if is_numeric_dtype(series) or is_bool_dtype(series) or is_datetime64_any_dtype(series):
        raise ValueError(f"{what}.{column} must hold text labels, got dtype {series.dtype}")
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.cat.remove_unused_categories()
        labels = pd.Series(series.cat.categories)
    else:
        labels = series
    kind = infer_dtype(labels, skipna=True)
    if kind not in ("string", "empty"):
        present = labels.dropna()
        bad = next((v for v in present if not isinstance(v, str)), None)
        raise ValueError(f"{what}.{column} must hold strings, found {bad!r} ({type(bad).__name__})")
    if isinstance(series.dtype, pd.CategoricalDtype):
        blank_cats = [c for c in series.cat.categories if not c.strip()]
        if blank_cats:
            if not allow_missing:
                raise ValueError(f"{what}.{column} has {int(series.isin(blank_cats).sum())} empty labels")
            series = series.cat.remove_categories(blank_cats)
    else:
        series = series.astype("str")
        blank = series.str.strip().eq("").fillna(False).astype(bool)
        if blank.any():
            if not allow_missing:
                raise ValueError(f"{what}.{column} has {int(blank.sum())} empty labels")
            series = series.mask(blank)
    missing = series.isna()
    if not allow_missing and missing.any():
        raise ValueError(f"{what}.{column} has {int(missing.sum())} missing values")
    return series


def validate_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Check a checkout-attempt frame; return a cleaned copy sorted by timestamp."""
    what = "transactions"
    tx = _require_frame(tx, what)
    _require_columns(tx, TRANSACTION_COLUMNS, what)
    if tx.empty:
        raise ValueError("transactions is empty; nothing to analyse")
    out = tx[TRANSACTION_COLUMNS].copy()
    out["timestamp"] = _naive_utc_times(out["timestamp"], "timestamp", what)

    ids = _label_column(out["transaction_id"], "transaction_id", what, allow_missing=False)
    if ids.duplicated().any():
        raise ValueError(f"{what}.transaction_id has {int(ids.duplicated().sum())} duplicate ids")
    out["transaction_id"] = ids.astype("str")

    for attr in SEGMENT_ATTRIBUTES:
        out[attr] = _label_column(out[attr], attr, what, allow_missing=False).astype("category")

    success = out["success"]
    if not (is_numeric_dtype(success) or is_bool_dtype(success)):
        raise ValueError(f"{what}.success must be 0/1 or boolean, got dtype {success.dtype}")
    if success.isna().any():
        raise ValueError(f"{what}.success has {int(success.isna().sum())} missing values")
    if not success.isin([0, 1]).all():
        bad = success[~success.isin([0, 1])].iloc[0]
        raise ValueError(f"{what}.success must be 0 or 1, found {bad!r}")
    out["success"] = success.astype(np.int64)

    for column in ("amount", "latency_ms"):
        values = out[column]
        if not is_numeric_dtype(values) or is_bool_dtype(values):
            raise ValueError(f"{what}.{column} must be numeric, got dtype {values.dtype}")
        values = values.astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{what}.{column} has missing or infinite values")
        if (values < 0).any():
            raise ValueError(f"{what}.{column} has {int((values < 0).sum())} negative values")
        out[column] = values

    codes = _label_column(out["error_code"], "error_code", what, allow_missing=False).astype("category")
    is_none = (codes == "NONE").to_numpy(dtype=bool)
    ok = out["success"].to_numpy() == 1
    mismatch = int((is_none != ok).sum())
    if mismatch:
        raise ValueError(
            f"{what}.error_code must be 'NONE' exactly for successful attempts; {mismatch} rows disagree with success"
        )
    out["error_code"] = codes

    if not out["timestamp"].is_monotonic_increasing:
        out = out.sort_values("timestamp", kind="stable")
    return out.reset_index(drop=True)


def validate_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    """Check a support-ticket frame; return a cleaned copy sorted by created_at.

    ``customer_country`` and ``device_os`` may be missing (None/NaN); blank strings
    are treated as missing. An empty frame is allowed (retrieval is then skipped).
    """
    what = "tickets"
    tickets = _require_frame(tickets, what)
    _require_columns(tickets, list(REQUIRED_TICKET_COLUMNS), what)
    columns = list(REQUIRED_TICKET_COLUMNS) + (["gt_incident_id"] if "gt_incident_id" in tickets.columns else [])
    out = tickets[columns].copy()
    if out.empty:
        log.warning("tickets is empty; ticket retrieval will be skipped")
    out["created_at"] = _naive_utc_times(out["created_at"], "created_at", what)

    ids = _label_column(out["ticket_id"], "ticket_id", what, allow_missing=False)
    if ids.duplicated().any():
        raise ValueError(f"{what}.ticket_id has {int(ids.duplicated().sum())} duplicate ids")
    out["ticket_id"] = ids.astype("str")
    out["channel"] = _label_column(out["channel"], "channel", what, allow_missing=False).astype("str")

    text = _label_column(out["text"], "text", what, allow_missing=False)
    out["text"] = text.astype("str")

    for column in _OPTIONAL_TICKET_TEXT:
        out[column] = _label_column(out[column], column, what, allow_missing=True).astype("str")
    if "gt_incident_id" in out.columns:
        out["gt_incident_id"] = _label_column(out["gt_incident_id"], "gt_incident_id", what, allow_missing=True).astype(
            "str"
        )

    if not out["created_at"].is_monotonic_increasing:
        out = out.sort_values("created_at", kind="stable")
    return out.reset_index(drop=True)
