"""The REES46 adapter: event log -> transaction schema, no tickets, incidents injected on a copy."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from kpi_rca.cli import main
from kpi_rca.data import Scenario, load_scenario, save_scenario, validate_transactions
from kpi_rca.data.adapters import (
    NO_PURCHASE,
    REES46_ATTRIBUTES,
    candidate_segments,
    inject_incident,
    no_tickets,
    rees46_transactions,
)


def events() -> pd.DataFrame:
    """Three sessions: s1 buys (two purchases), s2 abandons its cart, s3 has no category or brand."""
    rows = [
        ("2019-10-01 10:05:00", "cart", "electronics.smartphone", "apple", 900.0, "s1"),
        ("2019-10-01 10:06:00", "purchase", "electronics.smartphone", "apple", 900.0, "s1"),
        ("2019-10-01 10:07:00", "purchase", "electronics.audio", "apple", 100.0, "s1"),
        ("2019-10-01 10:01:00", "cart", "appliances.kitchen", "bosch", 50.0, "s2"),
        ("2019-10-01 11:00:00", "cart", None, None, 10.0, "s3"),
    ]
    frame = pd.DataFrame(rows, columns=["event_time", "event_type", "category_code", "brand", "price", "user_session"])
    frame["event_time"] = pd.to_datetime(frame["event_time"])
    return frame


def test_one_attempt_per_session_in_the_transaction_schema():
    tx = rees46_transactions(events()).set_index("transaction_id")
    assert list(tx.index) == ["s2", "s1", "s3"]  # time order of the first cart/purchase event
    assert tx.loc["s1", "success"] == 1 and tx.loc["s1", "amount"] == 1000.0 and tx.loc["s1", "error_code"] == "NONE"
    assert tx.loc["s2", "success"] == 0 and tx.loc["s2", "amount"] == 50.0 and tx.loc["s2", "error_code"] == NO_PURCHASE
    assert (tx.loc["s3", "category"], tx.loc["s3", "brand"]) == ("unknown", "unknown")
    assert tx.loc["s1", "category"] == "electronics"
    # passes the pipeline's own boundary check with the REES46 attributes, and keeps them
    clean = validate_transactions(tx.reset_index(), REES46_ATTRIBUTES)
    assert set(REES46_ATTRIBUTES) <= set(clean.columns) and "country" not in clean.columns


def test_brands_outside_the_top_n_become_other():
    extra = events().iloc[[0]].assign(user_session="s4")  # apple leads 2 to 1, no tie
    tx = rees46_transactions(pd.concat([events(), extra], ignore_index=True), top_brands=1)
    assert set(tx["brand"]) == {"apple", "other"}  # unknown (s3) is outside the top 1 as well


def test_missing_columns_or_no_events_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        rees46_transactions(events().drop(columns=["price"]))
    with pytest.raises(ValueError, match="empty"):
        rees46_transactions(events().iloc[:0])


def test_no_tickets_round_trips_as_an_empty_ticket_frame(tmp_path):
    tx = rees46_transactions(events())
    save_scenario(Scenario(transactions=tx, tickets=no_tickets(), ground_truth=None), tmp_path)
    loaded = load_scenario(tmp_path, attributes=REES46_ATTRIBUTES)
    assert loaded.tickets.empty and len(loaded.transactions) == 3
    with pytest.raises(ValueError, match="missing required columns"):
        load_scenario(tmp_path)  # the generator's attributes are not in this data


def _hourly(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(n)],
            "timestamp": pd.date_range("2019-10-01", periods=n, freq="min"),
            "brand": rng.choice(["apple", "samsung", "other"], n),
            "amount": 10.0,
            "success": 1,
            "error_code": "NONE",
        }
    )


def test_injection_changes_a_copy_and_only_the_segment_in_range():
    tx = _hourly()
    before = tx.copy()
    start, end = datetime(2019, 10, 1, 2), datetime(2019, 10, 1, 4)
    out, flipped = inject_incident(tx, {"brand": "apple"}, start, end, 1.0, np.random.default_rng(1))
    pd.testing.assert_frame_equal(tx, before)  # the original is untouched
    expected = (tx["brand"] == "apple") & (tx["timestamp"] >= start) & (tx["timestamp"] < end)
    assert (flipped == expected.to_numpy()).all() and flipped.sum() > 0
    assert (out.loc[flipped, "success"] == 0).all() and (out.loc[flipped, "error_code"] == NO_PURCHASE).all()
    assert (out.loc[~flipped, "success"] == 1).all()


def test_injection_rejects_bad_input():
    tx, rng = _hourly(), np.random.default_rng(1)
    start, end = datetime(2019, 10, 1, 2), datetime(2019, 10, 1, 4)
    with pytest.raises(ValueError, match="not in the data"):
        inject_incident(tx, {"country": "DE"}, start, end, 0.5, rng)
    with pytest.raises(ValueError, match="before its end"):
        inject_incident(tx, {"brand": "apple"}, end, start, 0.5, rng)
    with pytest.raises(ValueError, match="failure_rate"):
        inject_incident(tx, {"brand": "apple"}, start, end, 0.0, rng)


def test_candidate_segments_skip_other_and_respect_the_share_band():
    tx = _hourly()
    segments = candidate_segments(tx, ["brand"], lo=0.2, hi=0.5)
    assert {"brand": "other"} not in segments and {"brand": "apple"} in segments


def test_generate_and_evaluate_refuse_an_external_dataset_config(tmp_path, capsys):
    config = tmp_path / "external.yaml"
    config.write_text("dataset: external\nrca:\n  attributes: [brand]\n", encoding="utf-8")
    assert main(["generate", "--config", str(config), "--out-dir", str(tmp_path / "d")]) == 2
    assert main(["evaluate", "--config", str(config), "--out-dir", str(tmp_path / "o")]) == 2
    assert "generated data only" in capsys.readouterr().err
