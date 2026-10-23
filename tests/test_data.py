"""Synthetic generator, dataset I/O and boundary validation."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from kpi_rca.config import BACKGROUND_TICKET_THEMES, INCIDENT_TICKET_THEMES, config_from_dict
from kpi_rca.data import generate_scenario, load_scenario, save_scenario, validate_tickets, validate_transactions
from kpi_rca.data.generator import _BACKGROUND_CORE, _INCIDENT_CORE, TICKET_COLUMNS, TRANSACTION_COLUMNS

from conftest import small_overrides


def test_phrase_banks_cover_every_theme():
    assert set(_INCIDENT_CORE) == set(INCIDENT_TICKET_THEMES)
    assert set(_BACKGROUND_CORE) == set(BACKGROUND_TICKET_THEMES)


def test_generation_is_deterministic(small_cfg, small_scenario):
    again = generate_scenario(small_cfg.data, small_cfg.seed)
    pd.testing.assert_frame_equal(again.transactions, small_scenario.transactions)
    pd.testing.assert_frame_equal(again.tickets, small_scenario.tickets)
    other = generate_scenario(small_cfg.data, small_cfg.seed + 1)
    assert len(other.transactions) != len(small_scenario.transactions) or not other.transactions["success"].equals(
        small_scenario.transactions["success"]
    )


def test_transaction_schema(small_cfg, small_scenario):
    tx = small_scenario.transactions
    assert list(tx.columns) == TRANSACTION_COLUMNS
    assert tx["timestamp"].is_monotonic_increasing
    assert tx["timestamp"].min() >= pd.Timestamp(small_cfg.data.start)
    assert tx["timestamp"].max() < pd.Timestamp(small_cfg.data.end)
    assert tx["transaction_id"].is_unique
    assert set(tx["success"].unique()) <= {0, 1}
    assert ((tx["error_code"] == "NONE") == (tx["success"] == 1)).all()
    desktop = tx["device_type"] == "desktop"
    assert (tx.loc[desktop, "app_version"] == "web").all()
    assert (tx.loc[~desktop, "app_version"] != "web").all()
    assert set(tx.loc[desktop, "os"].unique()) <= {"Windows", "macOS"}
    assert (tx["amount"] > 0).all() and (tx["latency_ms"] > 0).all()
    # validation accepts generated data unchanged
    pd.testing.assert_frame_equal(validate_transactions(tx), tx)


def test_baseline_conversion_has_persistent_segment_effects(small_scenario):
    tx = small_scenario.transactions
    by_country = tx.groupby("country", observed=True)["success"].mean()
    assert 0.83 < tx["success"].mean() < 0.90
    assert by_country["BR"] < by_country["US"] - 0.08  # BR is always worse: normal, not an incident


def test_incident_forces_failures_only_inside_segment_and_time(small_scenario):
    tx = small_scenario.transactions
    inc = small_scenario.ground_truth.incidents[0]
    in_time = (tx["timestamp"] >= inc.start) & (tx["timestamp"] < inc.end)
    in_seg = (tx["country"] == "DE") & (tx["os"] == "iOS") & (tx["payment_gateway"] == "PayPal")
    coded = tx["error_code"] == inc.error_code
    assert coded.sum() == inc.forced_failures > 0
    assert not (coded & ~(in_time & in_seg)).any()
    fail_in = 1 - tx.loc[in_time & in_seg, "success"].mean()
    assert fail_in > 0.85
    fail_out = 1 - tx.loc[~in_time & in_seg, "success"].mean()
    assert fail_out < 0.15
    assert inc.segment_attempts == int((in_time & in_seg).sum())
    assert 0.02 < inc.segment_share < 0.07
    # forced failures are slow (latency multiplier)
    assert tx.loc[coded, "latency_ms"].median() > 3 * tx.loc[in_seg & ~in_time, "latency_ms"].median()



def test_true_loss_counts_only_orders_the_incident_destroyed(small_cfg, small_scenario):
    # Baseline outcomes are drawn before incidents, so the same seed without the incident
    # gives the counterfactual: the true loss is what succeeded there and failed here.
    clean = generate_scenario(small_cfg.data.model_copy(update={"incidents": []}), small_cfg.seed).transactions
    tx = small_scenario.transactions
    inc = small_scenario.ground_truth.incidents[0]
    lost = (clean["success"] == 1) & (tx["success"] == 0)
    assert inc.lost_orders == int(lost.sum()) > 0
    assert inc.lost_orders < inc.forced_failures  # some forced failures had failed anyway
    assert inc.lost_revenue == pytest.approx(tx.loc[lost, "amount"].sum(), abs=0.01)

def test_benign_surge_raises_volume_not_failures(small_scenario):
    tx = small_scenario.transactions
    ev = small_scenario.ground_truth.benign_events[0]
    us = tx[tx["country"] == "US"]
    during = us[(us["timestamp"] >= ev.start) & (us["timestamp"] < ev.end)]
    day_before = us[(us["timestamp"] >= ev.start - timedelta(days=1)) & (us["timestamp"] < ev.end - timedelta(days=1))]
    ratio = len(during) / len(day_before)
    assert 1.5 < ratio < 2.2
    assert abs(during["success"].mean() - day_before["success"].mean()) < 0.03
    assert ev.extra_attempts > 0


def test_ticket_schema_and_metadata_values(small_cfg, small_scenario):
    tk = small_scenario.tickets
    assert list(tk.columns) == TICKET_COLUMNS
    assert tk["ticket_id"].is_unique
    assert tk["created_at"].is_monotonic_increasing
    assert tk["created_at"].max() < pd.Timestamp(small_cfg.data.end)
    countries = set(tk["customer_country"].dropna().unique())
    assert countries <= set(small_cfg.data.countries)
    assert set(tk["device_os"].dropna().unique()) <= {"iOS", "Android", "Windows", "macOS"}
    assert tk["customer_country"].isna().any() and tk["device_os"].isna().any()
    for column in ("customer_country", "device_os", "channel", "gt_incident_id"):
        values = {str(v).lower() for v in tk[column].dropna().unique()}
        assert not values & {"unknown", "none", "nan", "n/a", ""}
    assert tk["text"].map(lambda s: s.isascii() and len(s) > 10).all()
    assert set(tk["channel"].unique()) <= set(small_cfg.data.tickets.channel_weights)
    pd.testing.assert_frame_equal(validate_tickets(tk), tk)


def test_incident_tickets_follow_failures(small_cfg, small_scenario):
    tk = small_scenario.tickets
    inc = small_scenario.ground_truth.incidents[0]
    own = tk[tk["gt_incident_id"] == inc.incident_id]
    assert len(own) == inc.tickets > 10
    assert (own["created_at"] >= inc.start).all()
    max_delay = timedelta(hours=small_cfg.data.tickets.delay_max_hours)
    assert (own["created_at"] < inc.end + max_delay).all()
    assert own["text"].str.contains("PayPal").mean() > 0.9
    assert own["text"].nunique() > len(own) * 0.5  # paraphrase variety
    known = own.dropna(subset=["customer_country"])
    assert (known["customer_country"] == "DE").all()
    known = own.dropna(subset=["device_os"])
    assert (known["device_os"] == "iOS").all()


def test_background_tickets_include_decoys(small_scenario):
    bg = small_scenario.tickets[small_scenario.tickets["gt_incident_id"].isna()]
    assert len(bg) > 100
    assert bg["text"].str.contains("PayPal").sum() >= 1  # PayPal refunds: same word, different problem
    assert bg["text"].str.contains("declined|refused", regex=True).sum() >= 5


def test_ground_truth_time_range(small_cfg, small_scenario):
    gt = small_scenario.ground_truth
    assert gt.seed == small_cfg.seed
    assert gt.start == small_cfg.data.start and gt.end == small_cfg.data.end
    assert [i.incident_id for i in gt.incidents] == ["INC-T1"]
    assert [b.event_id for b in gt.benign_events] == ["SURGE-T"]


def test_segment_filter_for_non_country_surge():
    over = small_overrides()
    over["data"]["benign_events"] = [
        {
            "event_id": "SURGE-ANDROID",
            "segment": {"os": "Android"},
            "start": "2026-09-05 10:00",
            "duration_hours": 3,
            "volume_multiplier": 2.0,
        }
    ]
    over["data"]["attempts_per_hour"] = 400
    cfg = config_from_dict(over)
    sc = generate_scenario(cfg.data, 3)
    tx = sc.transactions
    ev = sc.ground_truth.benign_events[0]
    in_time = (tx["timestamp"] >= ev.start) & (tx["timestamp"] < ev.end)
    before = (tx["timestamp"] >= ev.start - timedelta(days=1)) & (tx["timestamp"] < ev.end - timedelta(days=1))
    android = tx["os"] == "Android"
    assert 1.6 < (in_time & android).sum() / (before & android).sum() < 2.5
    assert 0.8 < (in_time & ~android).sum() / (before & ~android).sum() < 1.25


# ----------------------------------------------------------------------- I/O


def test_save_load_roundtrip(small_scenario, small_data_dir):
    loaded = load_scenario(small_data_dir)
    pd.testing.assert_frame_equal(loaded.transactions, small_scenario.transactions, check_categorical=False)
    pd.testing.assert_frame_equal(loaded.tickets, small_scenario.tickets)
    assert loaded.ground_truth == small_scenario.ground_truth


def test_load_without_ground_truth(tmp_path, small_scenario):
    sc = type(small_scenario)(small_scenario.transactions.head(500), small_scenario.tickets.head(20), None)
    paths = save_scenario(sc, tmp_path)
    assert "ground_truth" not in paths
    assert load_scenario(tmp_path).ground_truth is None


def test_load_missing_files_is_clear(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_scenario(tmp_path / "nope")
    with pytest.raises(ValueError, match="missing"):
        load_scenario(tmp_path)


# ---------------------------------------------------------------- validation


@pytest.fixture()
def tx_sample(small_scenario):
    return small_scenario.transactions.head(2000).copy()


@pytest.fixture()
def tk_sample(small_scenario):
    return small_scenario.tickets.head(50).copy()


def test_validate_transactions_rejects_bad_input(tx_sample):
    with pytest.raises(ValueError, match="DataFrame"):
        validate_transactions([1, 2])
    with pytest.raises(ValueError, match=r"missing required columns \['country'\]"):
        validate_transactions(tx_sample.drop(columns="country"))
    with pytest.raises(ValueError, match="empty"):
        validate_transactions(tx_sample.iloc[:0])

    bad = tx_sample.copy()
    bad["success"] = bad["success"].replace({0: 2})
    with pytest.raises(ValueError, match="0 or 1"):
        validate_transactions(bad)

    bad = tx_sample.copy()
    bad.loc[bad.index[0], "error_code"] = "CARD_DECLINED" if bad["success"].iloc[0] == 1 else "NONE"
    with pytest.raises(ValueError, match="'NONE' exactly"):
        validate_transactions(bad)

    bad = tx_sample.copy()
    bad["timestamp"] = bad["timestamp"].astype(str)
    with pytest.raises(ValueError, match="datetime"):
        validate_transactions(bad)

    bad = tx_sample.copy()
    bad.loc[bad.index[3], "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="missing timestamps"):
        validate_transactions(bad)

    bad = tx_sample.copy()
    bad["country"] = bad["country"].astype(object)
    bad.loc[bad.index[5], "country"] = None
    with pytest.raises(ValueError, match="country has 1 missing"):
        validate_transactions(bad)

    bad = tx_sample.copy()
    bad["os"] = 1
    with pytest.raises(ValueError, match="text labels"):
        validate_transactions(bad)

    bad = tx_sample.copy()
    bad.loc[bad.index[1], "transaction_id"] = bad["transaction_id"].iloc[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_transactions(bad)

    bad = tx_sample.copy()
    bad.loc[bad.index[2], "amount"] = -5.0
    with pytest.raises(ValueError, match="negative"):
        validate_transactions(bad)

    bad = tx_sample.copy()
    bad.loc[bad.index[2], "latency_ms"] = np.nan
    with pytest.raises(ValueError, match="latency_ms has missing"):
        validate_transactions(bad)


def test_validate_transactions_normalises(tx_sample):
    shuffled = tx_sample.sample(frac=1.0, random_state=0)
    shuffled["timestamp"] = shuffled["timestamp"].dt.tz_localize("Europe/Berlin")
    for attr in ("country", "os"):
        shuffled[attr] = shuffled[attr].astype(str)
    shuffled["success"] = shuffled["success"].astype(bool)
    out = validate_transactions(shuffled)
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].dt.tz is None
    assert out["success"].dtype == np.int64
    assert isinstance(out["country"].dtype, pd.CategoricalDtype)
    # Berlin local time converted to UTC: two hours earlier in September
    first_local = tx_sample["timestamp"].min()
    assert out["timestamp"].min() == first_local - timedelta(hours=2)


def test_validate_tickets(tk_sample):
    with pytest.raises(ValueError, match=r"missing required columns \['text'\]"):
        validate_tickets(tk_sample.drop(columns="text"))
    bad = tk_sample.copy()
    bad.loc[bad.index[0], "text"] = "   "
    with pytest.raises(ValueError, match="text has 1 empty"):
        validate_tickets(bad)
    bad = tk_sample.copy()
    bad.loc[bad.index[0], "text"] = np.nan
    with pytest.raises(ValueError, match="text has 1 missing"):
        validate_tickets(bad)
    bad = tk_sample.copy()
    bad.loc[bad.index[1], "ticket_id"] = bad["ticket_id"].iloc[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_tickets(bad)

    blank = tk_sample.copy()
    blank.loc[blank.index[0], "customer_country"] = ""
    out = validate_tickets(blank)
    assert pd.isna(out.loc[0, "customer_country"])

    no_gt = validate_tickets(tk_sample.drop(columns="gt_incident_id"))
    assert "gt_incident_id" not in no_gt.columns
    empty = validate_tickets(tk_sample.iloc[:0])
    assert empty.empty and list(empty.columns) == TICKET_COLUMNS
