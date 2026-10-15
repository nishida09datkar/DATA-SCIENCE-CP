"""Step 2, root-cause analysis: which segments explain an anomaly window's excess failures.

    analyze_window      M0 offset + M1 + TreeSHAP + SHAP-guided slice search (the method)
    brute_force_window  every slice up to brute_force_max_order attributes (the comparator)
"""

from kpi_rca.rca.baseline import brute_force_window
from kpi_rca.rca.search import analyze_window

__all__ = ["analyze_window", "brute_force_window"]
