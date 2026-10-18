"""Configuration models and YAML loading."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from kpi_rca.config import AppConfig, ConfigError, config_from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_valid():
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert [i.incident_id for i in cfg.data.incidents] == ["INC-001", "INC-002"]
    assert cfg.data.incidents[0].end == datetime(2026, 9, 10, 17, 0)
    assert cfg.data.end == datetime(2026, 9, 22)
    assert cfg.retrieval.top_k == 8 and cfg.llm.model == "claude-opus-5-5"


def test_default_narrator_is_the_free_template():
    # Free-testing policy: Claude is a paid API, so no default may select it, even with a key set.
    assert AppConfig().llm.provider == "template"
    assert load_config(ROOT / "config" / "default.yaml").llm.provider == "template"


def test_default_yaml_loads():
    cfg = load_config(ROOT / "config" / "default.yaml")
    defaults = AppConfig()
    assert cfg.data == defaults.data
    assert cfg.detection == defaults.detection
    assert cfg.rca == defaults.rca
    assert cfg.retrieval == defaults.retrieval
    assert cfg.llm == defaults.llm
    assert cfg.evaluation == defaults.evaluation


def test_yaml_override_deep_merges(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "seed: 5\n"
        "detection:\n  n_estimators: 50\n"
        "data:\n  countries: {US: 0.5, DE: 0.5}\n  country_os: {}\n  country_gateways: {}\n"
        "  incidents: []\n  benign_events: []\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.seed == 5
    assert cfg.detection.n_estimators == 50
    assert cfg.detection.period == 24  # untouched sibling keeps its default
    assert cfg.data.countries == {"US": 0.5, "DE": 0.5}
    assert cfg.data.incidents == []


def test_unknown_key_is_reported_with_its_path():
    with pytest.raises(ConfigError, match=r"detection\.n_estimatorz"):
        config_from_dict({"detection": {"n_estimatorz": 5}})


def test_missing_and_unparsable_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("data: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(bad)
    top = tmp_path / "list.yaml"
    top.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(top)


def test_incident_outside_data_range_is_rejected():
    with pytest.raises(ConfigError, match="INC-002 runs"):
        config_from_dict({"data": {"days": 14}})


def test_incident_validation():
    base = {
        "incident_id": "X",
        "segment": {"country": "DE"},
        "start": "2026-09-03 10:00",
        "duration_hours": 2,
        "failure_rate": 0.5,
        "error_code": "E",
    }
    ok = config_from_dict({"data": {"incidents": [base], "benign_events": []}})
    assert ok.data.incidents[0].ticket_themes == ["generic_payment_failure"]
    for change, message in [
        ({"segment": {"planet": "Mars"}}, "unknown attributes"),
        ({"segment": {"country": "XX"}}, "country='XX'"),
        ({"failure_rate": 1.5}, "failure_rate"),
        ({"error_code": "NONE"}, "real failure code"),
        ({"ticket_themes": ["weather"]}, "ticket_themes"),
    ]:
        with pytest.raises(ConfigError, match=message):
            config_from_dict({"data": {"incidents": [{**base, **change}], "benign_events": []}})
    with pytest.raises(ConfigError, match="duplicate"):
        config_from_dict({"data": {"incidents": [base, base], "benign_events": []}})


def test_timezone_aware_start_becomes_naive_utc():
    cfg = config_from_dict(
        {
            "data": {
                "start": "2026-09-01T02:00:00+02:00",
                "incidents": [],
                "benign_events": [],
            }
        }
    )
    assert cfg.data.start == datetime(2026, 9, 1, 0, 0)
    assert cfg.data.start.tzinfo is None


def test_profile_and_weight_validation():
    with pytest.raises(ConfigError, match="24 positive"):
        config_from_dict({"data": {"hourly_profile": [1.0] * 23}})
    with pytest.raises(ConfigError, match="must be > 0"):
        config_from_dict({"data": {"error_mix": {"CARD_DECLINED": 0.0}}})
    with pytest.raises(ConfigError, match="web"):
        config_from_dict({"data": {"app_versions": {"web": 1.0}}})
    with pytest.raises(ConfigError, match="stream_warmup_days"):
        config_from_dict({"detection": {"stream_warmup_days": 20, "stream_history_days": 10}})
    with pytest.raises(ConfigError, match="min_incidents"):
        config_from_dict({"evaluation": {"min_incidents": 3, "max_incidents": 2}})
    with pytest.raises(ConfigError, match="freq"):
        config_from_dict({"detection": {"freq": "every hour"}})
