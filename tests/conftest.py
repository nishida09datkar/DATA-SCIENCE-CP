"""Shared fixtures: one small generated scenario reused across the session.

Ten days at 1500 attempts/hour (~360k rows) with one incident and one benign surge,
large enough for detection and RCA to behave like the full-size scenario while
keeping the suite fast.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from kpi_rca.config import AppConfig, config_from_dict
from kpi_rca.data import Scenario, generate_scenario

SMALL_INCIDENT_START = datetime(2026, 9, 8, 13, 0)
SMALL_SURGE_START = datetime(2026, 9, 6, 18, 0)


def small_overrides() -> dict:
    return {
        "seed": 11,
        "data": {
            "days": 10,
            "attempts_per_hour": 1500,
            "incidents": [
                {
                    "incident_id": "INC-T1",
                    "segment": {"country": "DE", "os": "iOS", "payment_gateway": "PayPal"},
                    "start": SMALL_INCIDENT_START,
                    "duration_hours": 4,
                    "failure_rate": 0.9,
                    "error_code": "PAYPAL_SDK_TIMEOUT",
                    "latency_multiplier": 5.0,
                    "ticket_rate": 0.1,
                    "ticket_themes": ["paypal_timeout"],
                }
            ],
            "benign_events": [
                {
                    "event_id": "SURGE-T",
                    "segment": {"country": "US"},
                    "start": SMALL_SURGE_START,
                    "duration_hours": 5,
                    "volume_multiplier": 1.8,
                    "description": "test campaign",
                }
            ],
        },
        "reporting": {"make_plots": False},
    }


@pytest.fixture(scope="session")
def small_cfg() -> AppConfig:
    return config_from_dict(small_overrides(), source="tests/conftest.py")


@pytest.fixture(scope="session")
def small_scenario(small_cfg: AppConfig) -> Scenario:
    return generate_scenario(small_cfg.data, small_cfg.seed)


@pytest.fixture(scope="session")
def small_data_dir(tmp_path_factory, small_scenario: Scenario):
    from kpi_rca.data import save_scenario

    out = tmp_path_factory.mktemp("small_data")
    save_scenario(small_scenario, out)
    return out
