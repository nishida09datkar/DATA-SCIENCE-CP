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
