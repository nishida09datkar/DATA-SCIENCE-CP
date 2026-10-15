"""SHAP-guided segment search: turn per-row explanations into a few ranked causes.

For one anomaly window, with p0 from M0 and f(x) from M1 (see segment_model):

1. Excess. Each window row has excess_i = y_i - p0_i (y = 1 when the attempt failed).
   The window's total excess E = sum_i excess_i is the extra failure count to explain.
2. Mine. Take the failed rows that M1 pushes up (f(x) > 0), at most ``shap_max_rows`` of
   them by margin, and keep each row's active features whose SHAP value is clearly positive:
   phi >= max(shap_positive_abs, shap_positive_rel * the row's largest phi). Every subset of
   those features (up to ``max_order`` attributes) is a candidate slice; the
   ``top_k_patterns`` most frequent ones are scored. This replaces the brute-force
   enumeration of all slices (baseline.py) with a few dozen candidates.
3. Score. For a slice S (over the rows not yet explained by an earlier finding):
       EP     = sum_S excess / E                  share of the window's excess inside S
       purity = sum_S excess / sum_S (1 - p0)     how much of S's normal success was lost
       score  = 2 * EP * purity / (EP + purity)   harmonic mean, so S must be both big and pure
   A one-sided two-proportion z-test compares S's failure rate in the window with S's rate
   in the baseline period; S counts only if p < alpha / (slices scored this round), a
   Bonferroni correction. With fewer than ``min_segment_rows`` baseline rows the test uses
   the model instead: z = (fails - sum p0) / sqrt(sum p0 (1 - p0)).
4. Choose. Among eligible slices (n >= min_segment_rows, EP >= min_ep, significant) take
   the best score; slices within ``prune_tolerance`` of it count as ties and the one with
   the fewest attributes wins. Then prune: drop an attribute while the score stays within
   ``prune_tolerance``, so the cause is stated as simply as the data allows.
5. Repeat. Remove the chosen slice's rows and go back to step 2, until no slice is eligible
   or ``max_findings`` is reached. EP always divides by the original E, so the findings'
   EPs add up to the share of the window they explain together. EP can exceed 1 when the
   rest of the window happens to fail a little less than M0 expects (its excess is < 0).

Why the z-test against the baseline: BR fails more than other countries every day. Its
window failure rate equals its baseline rate, so BR is never significant, whatever M1 says.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from kpi_rca.config import RCAConfig
from kpi_rca.rca.segment_model import (
    RANDOM_STATE,
    WindowData,
    baseline_failure_probability,
    change_margin,
    fit_baseline_model,
    fit_change_model,
    prepare_window_data,
    shap_values,
    top_interactions,
)
from kpi_rca.schemas import AnomalyWindow, RCAResult, SegmentAttribute, SegmentFinding

log = logging.getLogger(__name__)

TOP_FEATURES = 20  # feature_importance keeps this many "attribute=value" entries
TOP_ERROR_CODES = 3

Items = tuple[tuple[int, int], ...]  # ((attribute index, value code), ...) sorted by attribute


@dataclass(frozen=True)
class SliceStats:
    """Everything the selection rule and the finding need about one slice."""

    items: Items
    n_window: int
    fails_window: int
    n_baseline: int
    fails_baseline: int
    excess: float  # sum_S (y - p0)
    ep: float
    purity: float
    score: float
    p_value: float


def one_sided_p_value(
    fails_w: float, n_w: float, fails_b: float, n_b: float, expected_fails: float, var_fails: float, min_rows: int
) -> float:
    """P(failure rate this much higher by chance): two-proportion z-test, window vs baseline.

    Falls back to the model-based test (observed vs sum p0) when the baseline has fewer
    than ``min_rows`` rows of this slice.
    """
    if n_w <= 0:
        return 1.0
    if n_b >= min_rows:
        pooled = (fails_w + fails_b) / (n_w + n_b)
        se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n_w + 1.0 / n_b))
        diff = fails_w / n_w - fails_b / n_b
    else:
        se = np.sqrt(var_fails)
        diff = fails_w - expected_fails
    if se <= 0:
        return 0.0 if diff > 0 else 1.0
    return float(norm.sf(diff / se))


def slice_stats(
    items: Items,
    *,
    n_w: float,
    fails_w: float,
    n_b: float,
    fails_b: float,
    excess: float,
    expected_success: float,
    expected_fails: float,
    var_fails: float,
    total_excess: float,
    cfg: RCAConfig,
) -> SliceStats:
    """Score one slice from its sums (shared by the SHAP search and the brute force)."""
    ep = excess / total_excess
    purity = excess / expected_success if expected_success > 0 else 0.0
    score = 2.0 * ep * purity / (ep + purity) if ep > 0 and purity > 0 else 0.0
    return SliceStats(
        items=items,
        n_window=int(n_w),
        fails_window=int(round(fails_w)),
        n_baseline=int(n_b),
        fails_baseline=int(round(fails_b)),
        excess=float(excess),
        ep=float(ep),
        purity=float(purity),
        score=float(score),
        p_value=one_sided_p_value(fails_w, n_w, fails_b, n_b, expected_fails, var_fails, cfg.min_segment_rows),
    )


def is_eligible(s: SliceStats, cfg: RCAConfig, n_tests: int) -> bool:
    return (
        s.n_window >= cfg.min_segment_rows
        and s.score > 0
        and s.ep >= cfg.min_ep
        and s.p_value < cfg.alpha / max(n_tests, 1)
    )


def choose_slice(candidates: Iterable[SliceStats], cfg: RCAConfig, n_tests: int) -> SliceStats | None:
    """Best eligible score; among near-ties (within prune_tolerance) the fewest attributes."""
    eligible = [s for s in candidates if is_eligible(s, cfg, n_tests)]
    if not eligible:
        return None
    best = max(s.score for s in eligible)
    ties = [s for s in eligible if s.score >= (1.0 - cfg.prune_tolerance) * best]
    return min(ties, key=lambda s: (len(s.items), -s.score, s.items))


class SliceEvaluator:
    """Scores slices of one window; ``active`` marks the rows no earlier finding explained."""

    def __init__(self, data: WindowData, p0: np.ndarray, cfg: RCAConfig):
        self.data = data
        self.cfg = cfg
        self.p0 = p0
        self.excess = data.y_window - p0
        self.total_excess = float(self.excess.sum())
        self.evaluated = 0

    def window_mask(self, items: Items, active: np.ndarray) -> np.ndarray:
        mask = active.copy()
        for a, code in items:
            mask &= self.data.window_codes[:, a] == code
        return mask

    def baseline_mask(self, items: Items) -> np.ndarray:
        mask = np.ones(len(self.data.y_baseline), dtype=bool)
        for a, code in items:
            mask &= self.data.baseline_codes[:, a] == code
        return mask

    def evaluate(self, items: Items, active: np.ndarray) -> SliceStats:
        self.evaluated += 1
        w = self.window_mask(items, active)
        b = self.baseline_mask(items)
        p0 = self.p0[w]
        return slice_stats(
            items,
            n_w=int(w.sum()),
            fails_w=float(self.data.y_window[w].sum()),
            n_b=int(b.sum()),
            fails_b=float(self.data.y_baseline[b].sum()),
            excess=float(self.excess[w].sum()),
            expected_success=float((1.0 - p0).sum()),
            expected_fails=float(p0.sum()),
            var_fails=float((p0 * (1.0 - p0)).sum()),
            total_excess=self.total_excess,
            cfg=self.cfg,
        )


def _active_feature_columns(data: WindowData) -> np.ndarray:
    """Offset of each attribute's first one-hot column (columns are grouped by attribute)."""
    sizes = [len(data.categories[a]) for a in data.attributes]
    return np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)


def mine_patterns(data: WindowData, rows: np.ndarray, phi: np.ndarray, cfg: RCAConfig) -> list[Items]:
    """Step 2: the ``top_k_patterns`` most frequent subsets of each row's SHAP-positive values.

    ``phi`` holds the SHAP values of ``rows`` (same order). Ties in frequency go to the
    slice with fewer attributes, then to the lowest codes, so the result is deterministic.
    """
    if len(rows) == 0:
        return []
    codes = data.window_codes[rows]
    n_attr = codes.shape[1]
    cols = _active_feature_columns(data)[None, :] + codes  # (n, n_attr): the row's active one-hot column
    row_phi = np.take_along_axis(phi, cols, axis=1)
    threshold = np.maximum(cfg.shap_positive_abs, cfg.shap_positive_rel * row_phi.max(axis=1, keepdims=True))
    positive = row_phi >= threshold
    bits = positive.astype(np.int64) @ (1 << np.arange(n_attr, dtype=np.int64))

    counts: dict[Items, int] = {}
    for m in range(1, 1 << n_attr):
        attrs = [a for a in range(n_attr) if m >> a & 1]
        if len(attrs) > cfg.max_order:
            continue
        hit = (bits & m) == m
        if not hit.any():
            continue
        values, freq = np.unique(codes[hit][:, attrs], axis=0, return_counts=True)
        for value, f in zip(values, freq):
            counts[tuple(zip(attrs, (int(v) for v in value)))] = int(f)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    return [items for items, _ in ranked[: cfg.top_k_patterns]]


def prune_slice(chosen: SliceStats, ev: SliceEvaluator, active: np.ndarray, n_tests: int) -> SliceStats:
    """Step 4: drop attributes while the score stays within prune_tolerance of the current one."""
    cfg = ev.cfg
    current = chosen
    while len(current.items) > 1:
        options = [ev.evaluate(current.items[:k] + current.items[k + 1 :], active) for k in range(len(current.items))]
        keep = [
            s for s in options if is_eligible(s, cfg, n_tests) and s.score >= (1.0 - cfg.prune_tolerance) * current.score
        ]
        if not keep:
            break
        current = max(keep, key=lambda s: (s.score, s.items))
    return current


def make_finding(
    rank: int, s: SliceStats, mask: np.ndarray, data: WindowData, m1: lgb.Booster | None, cfg: RCAConfig
) -> SegmentFinding:
    """SegmentFinding for slice ``s``; ``mask`` = its window rows at the time it was chosen."""
    attributes = [SegmentAttribute(attribute=a, value=v) for a, v in (data.item_label(i, c) for i, c in s.items)]
    contribution: dict[str, float] = {}
    if m1 is not None:
        rows = np.flatnonzero(mask)
        if len(rows) > cfg.shap_max_rows:
            rows = np.sort(np.random.default_rng(RANDOM_STATE).choice(rows, size=cfg.shap_max_rows, replace=False))
        phi = shap_values(m1, data, rows)
        for a, c in s.items:
            contribution[data.feature_names[data.column(a, c)]] = float(phi[:, data.column(a, c)].mean())
    failed = mask & (data.y_window == 1)
    errors = Counter(data.error_codes[failed].tolist()).most_common(TOP_ERROR_CODES)
    return SegmentFinding(
        rank=rank,
        attributes=attributes,
        ep=s.ep,
        purity=s.purity,
        score=s.score,
        failure_rate_window=s.fails_window / s.n_window if s.n_window else 0.0,
        failure_rate_baseline=s.fails_baseline / s.n_baseline if s.n_baseline else 0.0,
        n_window=s.n_window,
        n_baseline=s.n_baseline,
        excess_failures=s.excess,
        p_value=s.p_value,
        shap_contribution=contribution,
        top_error_codes={code: int(n) for code, n in errors},
    )


def empty_result(window: AnomalyWindow, method: str, started: float, evaluated: int = 0) -> RCAResult:
    return RCAResult(
        window_id=window.window_id,
        findings=[],
        feature_importance={},
        top_interactions=[],
        slices_evaluated=evaluated,
        runtime_s=time.perf_counter() - started,
        method=method,
    )


def feature_importance(m1: lgb.Booster, data: WindowData, cfg: RCAConfig) -> dict[str, float]:
    """Mean |SHAP| per one-hot feature over (a seeded sample of) the window rows, top 20."""
    n = len(data.y_window)
    rows = np.arange(n)
    if n > cfg.shap_max_rows:
        rows = np.sort(np.random.default_rng(RANDOM_STATE).choice(n, size=cfg.shap_max_rows, replace=False))
    mean_abs = np.abs(shap_values(m1, data, rows)).mean(axis=0)
    order = np.argsort(-mean_abs, kind="stable")[:TOP_FEATURES]
    return {data.feature_names[j]: float(mean_abs[j]) for j in order if mean_abs[j] > 0}


def analyze_window(
    transactions: pd.DataFrame,
    window: AnomalyWindow,
    cfg: RCAConfig,
    use_offset: bool = True,
    *,
    exclude_windows: Sequence[AnomalyWindow] = (),
) -> RCAResult:
    """Root causes of one anomaly window by SHAP-guided search (method "shap_search").

    ``use_offset=False`` is ablation A2 (no M0 offset). ``exclude_windows`` are other
    detected windows to keep out of the baseline period.
    """
    started = time.perf_counter()
    data = prepare_window_data(transactions, window, cfg, exclude_windows)
    if len(data.y_baseline) == 0:
        log.warning("rca: %s has no baseline rows in the %.1f days before it; no findings", window.window_id, cfg.baseline_days)
        return empty_result(window, "shap_search", started)

    m0 = fit_baseline_model(data, cfg) if use_offset else None
    p0 = baseline_failure_probability(data, m0, cfg, use_offset=use_offset)
    ev = SliceEvaluator(data, p0, cfg)
    if ev.total_excess <= 0 or data.y_window.min() == data.y_window.max():
        log.info("rca: %s has no excess failures (E=%.1f); no findings", window.window_id, ev.total_excess)
        return empty_result(window, "shap_search", started)

    m1 = fit_change_model(data, p0, cfg, use_offset=use_offset)
    margin = change_margin(m1, data)
    pushed_up = np.flatnonzero((data.y_window == 1) & (margin > 0))
    mined = pushed_up[np.argsort(-margin[pushed_up], kind="stable")[: cfg.shap_max_rows]]
    mined_phi = shap_values(m1, data, mined)

    active = np.ones(len(data.y_window), dtype=bool)
    findings: list[SegmentFinding] = []
    while len(findings) < cfg.max_findings:
        keep = active[mined]
        patterns = mine_patterns(data, mined[keep], mined_phi[keep], cfg)
        if not patterns:
            break
        candidates = [ev.evaluate(items, active) for items in patterns]
        chosen = choose_slice(candidates, cfg, n_tests=len(candidates))
        if chosen is None:
            break
        chosen = prune_slice(chosen, ev, active, n_tests=len(candidates))
        mask = ev.window_mask(chosen.items, active)
        findings.append(make_finding(len(findings) + 1, chosen, mask, data, m1, cfg))
        active &= ~mask

    interaction_rows = mined[: cfg.interaction_rows]
    result = RCAResult(
        window_id=window.window_id,
        findings=findings,
        feature_importance=feature_importance(m1, data, cfg),
        top_interactions=top_interactions(m1, data, interaction_rows),
        slices_evaluated=ev.evaluated,
        runtime_s=time.perf_counter() - started,
        method="shap_search",
    )
    log.info(
        "rca: %s shap_search -> %d finding(s) [%s] from %d slices in %.2fs",
        window.window_id,
        len(findings),
        "; ".join(f.label() for f in findings),
        ev.evaluated,
        result.runtime_s,
    )
    return result
