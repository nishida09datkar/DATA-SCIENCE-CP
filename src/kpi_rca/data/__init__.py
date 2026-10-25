"""Synthetic scenario generation and dataset I/O."""

from kpi_rca.data.generator import GroundTruth, Scenario, generate_scenario
from kpi_rca.data.io import load_scenario, save_scenario, validate_tickets, validate_transactions

__all__ = [
    "GroundTruth",
    "Scenario",
    "generate_scenario",
    "load_scenario",
    "save_scenario",
    "validate_tickets",
    "validate_transactions",
]

from kpi_rca.data.ecom_kaggle import load_kaggle_ecom_hourly, KaggleEcomLoader, build_kpi_frame_for_pipeline, validate_ecom_kpis  # noqa: F401

