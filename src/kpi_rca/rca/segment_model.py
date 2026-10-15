"""Excess-failure model: what is normal (M0) and what changed in the window (M1).

    M0    p0(x) = P(fail | x)                          LightGBM on the baseline period
    M1    logit P(fail | x, window) = logit p0(x) + f(x)
          LightGBM on the window rows with init_score = logit p0(x), so the trees
          fit only f(x), the change in failure log-odds during the window
    SHAP  f(x) = phi_0 + sum_j phi_j(x)                exact TreeSHAP of M1

Features are one-hot ``attribute=value`` columns, so every SHAP value reads as one
segment value (``country=DE``). LightGBM's ``pred_contrib`` computes TreeSHAP exactly;
it explains the trees only, and init_score is not part of the trees, so the values
explain f(x) and not the normal level p0.

Why the offset: BR fails more than any other country every day. A model of raw
failures would blame BR in every window; M0 already predicts BR's high failure rate,
so BR's excess y - p0 is near zero and M1 has nothing to learn there.

``use_offset=False`` is ablation A2: p0 becomes the baseline's overall failure rate for
every row and M1 is fitted from zero, so it explains raw failure log-odds.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from kpi_rca.config import RCAConfig
from kpi_rca.schemas import AnomalyWindow, InteractionPair

log = logging.getLogger(__name__)

RANDOM_STATE = 42  # baseline subsampling and LightGBM seeds; fixed so runs are reproducible
TOP_INTERACTIONS = 10


@dataclass
class WindowData:
    """The window rows and the baseline rows of one AnomalyWindow, as integer codes.

    ``*_codes[i, a]`` is the code of row i's value of ``attributes[a]``; the code
    indexes ``categories[attributes[a]]``. One-hot column j is the feature
    ``feature_names[j]`` = ``attributes[feature_attr[j]]=categories[...][feature_code[j]]``.
    """

    attributes: list[str]
    categories: dict[str, list[str]]
    window_codes: np.ndarray  # (n_window, n_attributes) int
    baseline_codes: np.ndarray  # (n_baseline, n_attributes) int
    y_window: np.ndarray  # 1 = the attempt failed
    y_baseline: np.ndarray
    error_codes: np.ndarray  # error_code of each window row (str)
    feature_names: list[str]
    feature_attr: np.ndarray
    feature_code: np.ndarray

    def one_hot(self, codes: np.ndarray) -> np.ndarray:
        return (codes[:, self.feature_attr] == self.feature_code[None, :]).astype(np.float32)

    def column(self, attribute_index: int, code: int) -> int:
        """One-hot column of ``(attribute, value code)``."""
        hits = np.flatnonzero((self.feature_attr == attribute_index) & (self.feature_code == code))
        return int(hits[0])

    def item_label(self, attribute_index: int, code: int) -> tuple[str, str]:
        attribute = self.attributes[attribute_index]
        return attribute, self.categories[attribute][code]


def prepare_window_data(
    transactions: pd.DataFrame,
    window: AnomalyWindow,
    cfg: RCAConfig,
    exclude_windows: Sequence[AnomalyWindow] = (),
) -> WindowData:
    """Cut the window rows and the ``cfg.baseline_days`` before the window out of ``transactions``.

    Other kpi_drop windows in ``exclude_windows`` are removed from the baseline, so an
    earlier incident does not teach M0 that its failures are normal.
    """
    missing = [c for c in ("timestamp", "success", "error_code", *cfg.attributes) if c not in transactions.columns]
    if missing:
        raise ValueError(f"transactions is missing columns {missing}")
    ts = transactions["timestamp"]
    in_window = ((ts >= window.start) & (ts < window.end)).to_numpy()
    baseline_start = window.start - timedelta(days=cfg.baseline_days)
    in_baseline = ((ts >= baseline_start) & (ts < window.start)).to_numpy()
    for other in exclude_windows:
        if other.kpi_type == "kpi_drop" and other.window_id != window.window_id:
            # Not `&=`: under pandas Copy-on-Write, .to_numpy() can return a read-only view.
            in_baseline = in_baseline & ~((ts >= other.start) & (ts < other.end)).to_numpy()
    if not in_window.any():
        raise ValueError(f"window {window.window_id} ({window.start}..{window.end}) contains no transactions")

    win = transactions.loc[in_window]
    base = transactions.loc[in_baseline]
    attributes = list(cfg.attributes)
    categories: dict[str, list[str]] = {}
    window_codes = np.empty((len(win), len(attributes)), dtype=np.int64)
    baseline_codes = np.empty((len(base), len(attributes)), dtype=np.int64)
    for a, attribute in enumerate(attributes):
        values = np.concatenate([win[attribute].astype("str").to_numpy(), base[attribute].astype("str").to_numpy()])
        cats, codes = np.unique(values, return_inverse=True)
        categories[attribute] = [str(c) for c in cats]
        window_codes[:, a] = codes[: len(win)]
        baseline_codes[:, a] = codes[len(win) :]

    feature_attr = np.concatenate([np.full(len(categories[a]), i) for i, a in enumerate(attributes)])
    feature_code = np.concatenate([np.arange(len(categories[a])) for a in attributes])
    feature_names = [f"{a}={v}" for a in attributes for v in categories[a]]
    return WindowData(
        attributes=attributes,
        categories=categories,
        window_codes=window_codes,
        baseline_codes=baseline_codes,
        y_window=(win["success"].to_numpy() == 0).astype(np.int64),
        y_baseline=(base["success"].to_numpy() == 0).astype(np.int64),
        error_codes=win["error_code"].astype("str").to_numpy(),
        feature_names=feature_names,
        feature_attr=feature_attr.astype(np.int64),
        feature_code=feature_code.astype(np.int64),
    )


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _params(cfg: RCAConfig) -> dict:
    params = {
        "objective": "binary",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "lambda_l2": cfg.lambda_l2,
        "seed": RANDOM_STATE,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    if cfg.num_threads:
        params["num_threads"] = cfg.num_threads
    return params


def fit_baseline_model(data: WindowData, cfg: RCAConfig) -> lgb.Booster | None:
    """M0 on (a seeded sample of at most ``max_baseline_rows``) baseline rows.

    Returns None when the baseline cannot be modelled (fewer than ``min_segment_rows``
    rows or only one class); p0 then falls back to the overall baseline failure rate.
    """
    n = len(data.y_baseline)
    if n < cfg.min_segment_rows or data.y_baseline.min() == data.y_baseline.max():
        log.warning("rca: baseline has %d rows / one class only; M0 falls back to a constant rate", n)
        return None
    rows = np.arange(n)
    if n > cfg.max_baseline_rows:
        rows = np.sort(np.random.default_rng(RANDOM_STATE).choice(n, size=cfg.max_baseline_rows, replace=False))
    train = lgb.Dataset(data.one_hot(data.baseline_codes[rows]), label=data.y_baseline[rows])
    return lgb.train(_params(cfg), train, num_boost_round=cfg.m0_rounds)


def baseline_failure_probability(
    data: WindowData, m0: lgb.Booster | None, cfg: RCAConfig, *, use_offset: bool = True
) -> np.ndarray:
    """p0 for every window row, clipped to [p0_clip, 1 - p0_clip] so its logit is finite."""
    if use_offset and m0 is not None:
        p0 = m0.predict(data.one_hot(data.window_codes))
    else:
        rate = float(data.y_baseline.mean()) if len(data.y_baseline) else float(data.y_window.mean())
        p0 = np.full(len(data.y_window), rate)
    return np.clip(np.asarray(p0, dtype=float), cfg.p0_clip, 1.0 - cfg.p0_clip)


def fit_change_model(data: WindowData, p0: np.ndarray, cfg: RCAConfig, *, use_offset: bool = True) -> lgb.Booster:
    """M1 on the window rows; with the offset its trees learn only f(x), the change in log-odds."""
    init = _logit(p0) if use_offset else None
    train = lgb.Dataset(data.one_hot(data.window_codes), label=data.y_window, init_score=init)
    return lgb.train(_params(cfg), train, num_boost_round=cfg.m1_rounds)


def change_margin(m1: lgb.Booster, data: WindowData, rows: np.ndarray | None = None) -> np.ndarray:
    """f(x): M1's raw tree output (without init_score) for the window rows."""
    codes = data.window_codes if rows is None else data.window_codes[rows]
    return np.asarray(m1.predict(data.one_hot(codes), raw_score=True), dtype=float)


def shap_values(m1: lgb.Booster, data: WindowData, rows: np.ndarray) -> np.ndarray:
    """Exact TreeSHAP of f(x) for the given window rows: (len(rows), n_features).

    The bias column phi_0 is dropped; row sums plus phi_0 equal ``change_margin``.
    """
    if len(rows) == 0:
        return np.zeros((0, len(data.feature_names)))
    contrib = m1.predict(data.one_hot(data.window_codes[rows]), pred_contrib=True)
    return np.asarray(contrib, dtype=float)[:, :-1]


def top_interactions(m1: lgb.Booster, data: WindowData, rows: np.ndarray, top: int = TOP_INTERACTIONS) -> list[InteractionPair]:
    """Strongest SHAP interaction pairs: mean |phi_ij| over ``rows`` (at most interaction_rows of them).

    Interaction values show which segment values act together, e.g. country=DE with
    payment_gateway=PayPal. They are reported, not used for the search. If the shap
    package cannot explain this booster, the list is empty and a warning is logged.
    """
    if len(rows) == 0:
        return []
    try:
        import shap

        explainer = shap.TreeExplainer(m1)
        values = np.asarray(explainer.shap_interaction_values(data.one_hot(data.window_codes[rows])), dtype=float)
    except Exception as exc:  # shap/LightGBM version mismatches must not stop the pipeline
        log.warning("rca: SHAP interaction values unavailable (%s: %s)", type(exc).__name__, exc)
        return []
    if values.ndim == 4:  # some shap versions return one array per class
        values = values[-1]
    strength = np.abs(values).mean(axis=0)
    i_idx, j_idx = np.triu_indices(strength.shape[0], k=1)
    pair_strength = strength[i_idx, j_idx]
    order = np.argsort(-pair_strength, kind="stable")[:top]
    return [
        InteractionPair(
            feature_a=data.feature_names[i_idx[k]],
            feature_b=data.feature_names[j_idx[k]],
            strength=float(pair_strength[k]),
        )
        for k in order
        if pair_strength[k] > 0
    ]
