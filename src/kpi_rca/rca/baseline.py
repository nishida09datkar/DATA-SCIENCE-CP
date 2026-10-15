"""Brute-force comparator: score every slice up to ``brute_force_max_order`` attributes.

This is the reference the SHAP-guided search is evaluated against. It uses the same M0
offset, the same score (harmonic mean of EP and purity), the same z-test and the same
selection rule (search.choose_slice), but no M1 and no mining: every combination of
1..brute_force_max_order attributes and every value combination present in the window is
scored. ``slices_evaluated`` counts those groups, so the evaluation can compare search
cost (slices scored, runtime) as well as accuracy.

Sums per slice are vectorised: for one attribute subset the value codes are packed into a
single integer key (np.ravel_multi_index), and np.bincount over that key gives every group's
n, failures, excess, sum(1 - p0), sum p0 and sum p0(1 - p0) in one pass.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from kpi_rca.config import RCAConfig
from kpi_rca.rca.search import SliceEvaluator, SliceStats, choose_slice, empty_result, make_finding, slice_stats
from kpi_rca.rca.segment_model import (
    WindowData,
    baseline_failure_probability,
    fit_baseline_model,
    prepare_window_data,
)
from kpi_rca.schemas import AnomalyWindow, RCAResult, SegmentFinding

log = logging.getLogger(__name__)


def _score_subset(
    attrs: tuple[int, ...], data: WindowData, ev: SliceEvaluator, active: np.ndarray, cfg: RCAConfig
) -> list[SliceStats]:
    """Every value combination of ``attrs`` that has active window rows."""
    dims = tuple(len(data.categories[data.attributes[a]]) for a in attrs)
    size = int(np.prod(dims))
    rows = np.flatnonzero(active)
    key_w = np.ravel_multi_index(tuple(data.window_codes[rows][:, attrs].T), dims)
    key_b = np.ravel_multi_index(tuple(data.baseline_codes[:, attrs].T), dims)
    p0 = ev.p0[rows]

    def sums(weights: np.ndarray | None, key: np.ndarray = key_w) -> np.ndarray:
        return np.bincount(key, weights=weights, minlength=size)

    n_w = sums(None)
    fails_w = sums(data.y_window[rows].astype(float))
    excess = sums(ev.excess[rows])
    exp_success = sums(1.0 - p0)
    exp_fails = sums(p0)
    var_fails = sums(p0 * (1.0 - p0))
    n_b = sums(None, key_b)
    fails_b = sums(data.y_baseline.astype(float), key_b)

    out = []
    for g in np.flatnonzero(n_w > 0):
        values = np.unravel_index(g, dims)
        items = tuple(zip(attrs, (int(v) for v in values)))
        out.append(
            slice_stats(
                items,
                n_w=n_w[g],
                fails_w=fails_w[g],
                n_b=n_b[g],
                fails_b=fails_b[g],
                excess=excess[g],
                expected_success=exp_success[g],
                expected_fails=exp_fails[g],
                var_fails=var_fails[g],
                total_excess=ev.total_excess,
                cfg=cfg,
            )
        )
    return out


def brute_force_window(
    transactions: pd.DataFrame,
    window: AnomalyWindow,
    cfg: RCAConfig,
    use_offset: bool = True,
    *,
    exclude_windows: Sequence[AnomalyWindow] = (),
) -> RCAResult:
    """Root causes of one anomaly window by exhaustive slice scoring (method "brute_force")."""
    started = time.perf_counter()
    data = prepare_window_data(transactions, window, cfg, exclude_windows)
    if len(data.y_baseline) == 0:
        log.warning("rca: %s has no baseline rows in the %.1f days before it; no findings", window.window_id, cfg.baseline_days)
        return empty_result(window, "brute_force", started)

    m0 = fit_baseline_model(data, cfg) if use_offset else None
    p0 = baseline_failure_probability(data, m0, cfg, use_offset=use_offset)
    ev = SliceEvaluator(data, p0, cfg)
    if ev.total_excess <= 0:
        log.info("rca: %s has no excess failures (E=%.1f); no findings", window.window_id, ev.total_excess)
        return empty_result(window, "brute_force", started)

    n_attr = len(data.attributes)
    subsets = [c for r in range(1, min(cfg.brute_force_max_order, n_attr) + 1) for c in combinations(range(n_attr), r)]
    active = np.ones(len(data.y_window), dtype=bool)
    findings: list[SegmentFinding] = []
    evaluated = 0
    while len(findings) < cfg.max_findings and active.any():
        candidates = [s for attrs in subsets for s in _score_subset(attrs, data, ev, active, cfg)]
        evaluated += len(candidates)
        chosen = choose_slice(candidates, cfg, n_tests=len(candidates))
        if chosen is None:
            break
        mask = ev.window_mask(chosen.items, active)
        findings.append(make_finding(len(findings) + 1, chosen, mask, data, None, cfg))
        active &= ~mask

    result = RCAResult(
        window_id=window.window_id,
        findings=findings,
        feature_importance={},
        top_interactions=[],
        slices_evaluated=evaluated,
        runtime_s=time.perf_counter() - started,
        method="brute_force",
    )
    log.info(
        "rca: %s brute_force -> %d finding(s) [%s] from %d slices in %.2fs",
        window.window_id,
        len(findings),
        "; ".join(f.label() for f in findings),
        evaluated,
        result.runtime_s,
    )
    return result
