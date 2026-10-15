"""KPI root-cause analysis engine.

Deterministic ML does the math (STL + Isolation Forest detection, LightGBM + SHAP
segment isolation, FAISS ticket retrieval) before a schema-guarded LLM writes a
short incident memo.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
