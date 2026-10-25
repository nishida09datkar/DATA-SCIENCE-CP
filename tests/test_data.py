"""Synthetic generator, dataset I/O and boundary validation."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from kpi_rca.config import BACKGROUND_TICKET_THEMES, INCIDENT_TICKET_THEMES, AppConfig, config_from_dict
from kpi_rca.data import generate_scenario, load_scenario, save_scenario, validate_tickets, validate_transactions
from kpi_rca.data.ecom_kaggle import (
    InsufficientEcomDataError,
    InvalidEcomDataError,
    KaggleEcomLoader,
    build_kpi_frame_for_pipeline,
    load_kaggle_ecom_hourly,
    validate_ecom_kpis,
)
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


# ------------------------------------------- injected anomaly controllability


def _single_incident_cfg(failure_rate: float, duration_hours: float = 4.0) -> AppConfig:
    """A tiny sealed scenario whose only injected anomaly is one DE failure incident."""
    return config_from_dict(
        {
            "seed": 3,
            "data": {
                "days": 4,
                "attempts_per_hour": 300,
                "incidents": [
                    {
                        "incident_id": "INC-CTL",
                        "segment": {"country": "DE"},
                        "start": "2026-09-02 10:00",
                        "duration_hours": duration_hours,
                        "failure_rate": failure_rate,
                        "error_code": "PAYPAL_SDK_TIMEOUT",
                    }
                ],
                "benign_events": [],
                "tickets": {"background_per_hour": 0.5},
            },
        },
        source="tests/test_data.py",
    )


def _incident_outcome(cfg) -> tuple[float, int]:
    """(success rate of the DE segment inside the incident window, forced-failure count)."""
    scenario = generate_scenario(cfg.data, cfg.seed)
    tx = scenario.transactions
    [inc] = scenario.ground_truth.incidents
    inside = (tx["timestamp"] >= inc.start) & (tx["timestamp"] < inc.end) & (tx["country"] == "DE")
    return float(1 - tx.loc[inside, "success"].mean()), inc.forced_failures


def test_injected_drop_magnitude_is_controllable():
    strong = _incident_outcome(_single_incident_cfg(failure_rate=0.9))
    weak = _incident_outcome(_single_incident_cfg(failure_rate=0.3))
    assert strong[1] > weak[1] > 0  # ground truth carries the injected counts
    # the stronger incident costs the segment more successes, with margin both ways
    assert strong[0] > 0.5 and weak[0] < strong[0] - 0.3


def test_injected_drop_duration_is_controllable():
    short = generate_scenario(_single_incident_cfg(0.9, duration_hours=2).data, 3)
    long = generate_scenario(_single_incident_cfg(0.9, duration_hours=6).data, 3)
    for scenario, hours in ((short, 2), (long, 6)):
        tx = scenario.transactions
        [inc] = scenario.ground_truth.incidents
        coded = tx["error_code"] == inc.error_code
        affected = (tx["timestamp"] >= inc.start) & (tx["timestamp"] < inc.end + timedelta(hours=1))
        assert inc.forced_failures > 0
        assert coded.sum() == inc.forced_failures
        assert (tx.loc[coded, "timestamp"] < inc.start + timedelta(hours=hours)).all()
        assert affected[coded].all()  # every forced failure lies inside the injected window


def test_injected_anomalies_are_seeded():
    first, second = (
        _incident_outcome(_single_incident_cfg(failure_rate=0.9)),
        _incident_outcome(_single_incident_cfg(failure_rate=0.9)),
    )
    assert first == second  # same seed -> identical ground truth down to the counts


# ----------------------------------- Kaggle eCommerce CSV (chunked, tiny fixture)

ECOM_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]


def _ecom_rows() -> list[tuple]:
    """Five hours with a duplicate pair, a cross-chunk session, nulls and one missing hour."""
    return [
        ("2019-11-01 00:00:00 UTC", "view", "100", "1", "electronics.smartphone", "xiaomi", 100.0, "u1", "s1"),
        ("2019-11-01 00:00:00 UTC", "purchase", "100", "1", "electronics.smartphone", "xiaomi", 100.0, "u1", "s1"),
        # exact duplicate of the purchase above (same chunk at chunksize=3)
        ("2019-11-01 00:00:00 UTC", "purchase", "100", "1", "electronics.smartphone", "xiaomi", 100.0, "u1", "s1"),
        # same session as rows 0-2 but in the next chunk: must not inflate the hourly session count
        ("2019-11-01 00:00:00 UTC", "view", "100", "1", "electronics.smartphone", "xiaomi", 100.0, "u1", "s1"),
        ("2019-11-01 00:00:00 UTC", "cart", "200", "2", "electronics.laptop", "lenovo", 300.0, "u2", "s2"),
        ("2019-11-01 01:00:00 UTC", "purchase", "200", "2", "electronics.laptop", "lenovo", 300.0, "u2", "s2"),
        ("2019-11-01 01:00:00 UTC", "view", "300", "3", "appliances.sewing_machine", "janome", 50.0, "u3", "s3"),
        # 02:00 is absent -> imputed hour
        ("2019-11-01 03:00:00 UTC", "view", "400", "4", "", "", 10.0, "u4", "s4"),
        ("2019-11-01 03:00:00 UTC", "purchase", "400", "4", "", "", 10.0, "u4", "s4"),
        ("2019-11-01 04:00:00 UTC", "view", "500", "5", "electronics.smartphone", "otherbrand", 20.0, "u5", "s5"),
        ("2019-11-01 04:00:00 UTC", "view", "500", "5", "electronics.smartphone", "otherbrand", 20.0, "u5", "s5"),
    ]


def _write_ecom_csv(tmp_path, rows: list[tuple], name: str = "ecom.csv") -> "Path":
    path = tmp_path / name
    pd.DataFrame(rows, columns=ECOM_COLUMNS).to_csv(path, index=False)
    return path


def _load_fixture(tmp_path, *, chunksize: int = 3, cache_dir=None):
    csv = _write_ecom_csv(tmp_path, _ecom_rows())
    return KaggleEcomLoader(csv, cache_dir=cache_dir, chunksize=chunksize, top_brands=2).load()


def test_ecom_chunked_hourly_aggregation_is_exact(tmp_path):
    dataset = _load_fixture(tmp_path)
    hourly = dataset.hourly
    assert isinstance(hourly.index, pd.DatetimeIndex) and hourly.index.tz is None
    assert hourly.index.name == "hour"
    assert len(hourly) == 5
    assert hourly.index.tolist() == list(pd.date_range("2019-11-01 00:00", periods=5, freq="h"))

    h00 = hourly.loc["2019-11-01 00:00"]
    assert (h00["sessions"], h00["purchases"], h00["purchase_events"], h00["carts"]) == (2, 1, 1, 1)
    assert h00["revenue"] == 100.0
    assert h00["conversion_rate"] == 0.5 and h00["cart_rate"] == 0.5

    h01 = hourly.loc["2019-11-01 01:00"]
    assert (h01["sessions"], h01["purchases"], h01["carts"], h01["revenue"]) == (2, 1, 0, 300.0)
    assert h01["conversion_rate"] == 0.5 and h01["cart_rate"] == 0.0

    h04 = hourly.loc["2019-11-01 04:00"]
    assert (h04["sessions"], h04["purchases"], h04["purchase_events"]) == (1, 0, 0)


def test_ecom_missing_hour_is_imputed_and_reported(tmp_path):
    dataset = _load_fixture(tmp_path)
    missing = pd.Timestamp("2019-11-01 02:00")
    assert dataset.hourly.loc[missing, "imputed"]
    assert dataset.hourly.loc[missing, "sessions"] == 0
    assert dataset.hourly.loc[missing, "purchases"] == 0
    assert np.isfinite(dataset.hourly.loc[missing, "conversion_rate"])  # interpolated, not NaN
    observed = dataset.hourly.drop(index=missing)
    assert not observed["imputed"].any()
    assert dataset.gaps == [{"start": "2019-11-01T02:00:00", "end": "2019-11-01T03:00:00", "hours": 1}]


def test_ecom_null_category_and_brand_become_unknown(tmp_path):
    dataset = _load_fixture(tmp_path)
    h03 = dataset.segments[(dataset.segments["hour"] == pd.Timestamp("2019-11-01 03:00"))]
    category = h03[(h03["attribute"] == "category") & (h03["value"] == "unknown")].iloc[0]
    brand = h03[(h03["attribute"] == "brand") & (h03["value"] == "unknown")].iloc[0]
    assert (category["sessions"], category["purchases"], category["revenue"]) == (1, 1, 10.0)
    assert (brand["sessions"], brand["purchases"], brand["revenue"]) == (1, 1, 10.0)


def test_ecom_top_brands_plus_other(tmp_path):
    dataset = _load_fixture(tmp_path)
    assert set(dataset.top_brands) == {"xiaomi", "lenovo"}  # top-2 by raw event count
    brands = set(dataset.segments.loc[dataset.segments["attribute"] == "brand", "value"])
    assert brands == {"xiaomi", "lenovo", "unknown", "other"}
    other_h01 = dataset.segments[
        (dataset.segments["attribute"] == "brand")
        & (dataset.segments["value"] == "other")
        & (dataset.segments["hour"] == pd.Timestamp("2019-11-01 01:00"))
    ].iloc[0]
    assert other_h01["sessions"] == 1  # janome, outside the top-2, folded into "other"


def test_ecom_cache_roundtrip(tmp_path):
    cache = tmp_path / "processed"
    csv = _write_ecom_csv(tmp_path, _ecom_rows())
    first = KaggleEcomLoader(csv, cache_dir=cache, chunksize=3, top_brands=2).load()
    assert first.cache_paths and all(path.is_file() for path in first.cache_paths.values())
    second = KaggleEcomLoader(csv, cache_dir=cache, chunksize=3, top_brands=2).load()
    pd.testing.assert_frame_equal(first.hourly, second.hourly)
    pd.testing.assert_frame_equal(first.segments, second.segments)
    assert first.gaps == second.gaps


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("2019-11-01 00:00:00 UTC", "purchase", "1", "1", "electronics", "x", -5.0, "u", "s"), "negative"),
        (("2019-11-01 00:00:00 UTC", "refund", "1", "1", "electronics", "x", 5.0, "u", "s"), "event_type"),
        (("not-a-timestamp", "view", "1", "1", "electronics", "x", 5.0, "u", "s"), "timestamps"),
        (("2019-11-01 00:00:00 UTC", "purchase", "1", "1", "electronics", "x", "", "u", "s"), "price"),
    ],
)
def test_ecom_invalid_rows_raise_typed_errors(tmp_path, row, message):
    csv = _write_ecom_csv(tmp_path, [row])
    with pytest.raises(InvalidEcomDataError, match=message):
        KaggleEcomLoader(csv, cache_dir=None, chunksize=2).load()


def test_ecom_validation_rejects_short_and_gapped_frames():
    index = pd.date_range("2019-11-01", periods=10, freq="h", name="hour")
    frame = pd.DataFrame(
        {
            "sessions": np.full(10, 10),
            "purchases": np.full(10, 2),
            "conversion_rate": np.full(10, 0.2),
            "cart_rate": np.full(10, 0.3),
            "revenue": np.full(10, 100.0),
            "imputed": False,
        },
        index=index,
    )
    with pytest.raises(InsufficientEcomDataError, match="two full seasonal periods"):
        validate_ecom_kpis(frame, period=24)
    with pytest.raises(InvalidEcomDataError, match="regular"):
        validate_ecom_kpis(frame.drop(index=index[4]), period=2)
    bad = frame.copy()
    bad.loc[index[0], "purchases"] = 11
    with pytest.raises(InvalidEcomDataError, match="purchases greater than sessions"):
        validate_ecom_kpis(bad, period=2)
    assert validate_ecom_kpis(frame, period=2).index.name == "hour"


def test_ecom_maps_to_pipeline_kpi_columns(tmp_path):
    csv = _write_ecom_csv(tmp_path, _ecom_rows())
    hourly = load_kaggle_ecom_hourly(csv, cache_dir=None, chunksize=3, top_brands=2, period=2)
    kpi = build_kpi_frame_for_pipeline(hourly, period=2)
    assert list(kpi.columns) == [
        "attempts",
        "successes",
        "failures",
        "conversion_rate",
        "revenue",
        "cart_rate",
        "imputed",
    ]
    assert kpi.index.name == "bucket"
    h00 = kpi.loc["2019-11-01 00:00"]
    assert (h00["attempts"], h00["successes"], h00["failures"]) == (2, 1, 1)
    assert h00["revenue"] == 100.0

    from kpi_rca.detection.kpi import preprocess_kpis

    clean = preprocess_kpis(kpi)  # the detection stage accepts the mapped frame unchanged
    assert len(clean) == len(kpi) and clean["imputed"].sum() == 1
    assert (clean["attempts"] >= clean["successes"]).all()

