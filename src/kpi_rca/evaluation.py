"""The randomised benchmark of PLAN section 5: ``python -m kpi_rca evaluate``.

For every scenario s = 0 .. n-1:

1. Sample 1-2 incidents and one benign surge (seeded by evaluation.seed and s):
   - the segment comes from a catalogue built on a short incident-free pilot run, so its
     traffic share is known in advance and lies in [min_segment_share, max_segment_share];
   - redundant segments are left out of the catalogue, so the ground truth is never
     ambiguous: {os=Windows, device_type=desktop} is the same rows as {os=Windows}, and
     app_version=web is the same rows as device_type=desktop;
   - the failure rate is set from a target visibility: the drop in GLOBAL conversion,
     measured in hourly binomial standard deviations (evaluation.visibility_sigma);
   - the error code and ticket theme follow the segment (PayPal -> PayPal timeout, ...);
   - events are at least MIN_GAP_H apart, so no surge overlaps an incident.
2. Generate the scenario and run each stage the way the pipeline does:
   batch detection, streaming delay, RCA (guided, brute force, and without the offset),
   retrieval per backend with and without the metadata boosts, and the template memo
   through the guardrails.
3. Score each stage against the ground truth the generator wrote.

Extra incident-free scenarios (evaluation.incident_free_runs) measure false alarms per
day. Every number is deterministic for a given config and seed, except runtimes.

Outputs in ``<out_dir>/evaluation/``: metrics.json (everything), summary.md (the results
table against the PLAN 5.4 targets) and scenarios.csv (one row per injected incident).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from kpi_rca.config import WEB_APP_VERSION, AppConfig, BenignEventSpec, DataConfig, IncidentSpec
from kpi_rca.data import Scenario, generate_scenario
from kpi_rca.detection import StreamingDetector, aggregate_kpis, detect_anomalies, detection_delays
from kpi_rca.detection.streaming import bucket_step
from kpi_rca.pipeline import GROUND_TRUTH_TICKET_COLUMNS
from kpi_rca.rca import analyze_window, brute_force_window
from kpi_rca.reporting.context import build_incident_context
from kpi_rca.reporting.impact import compute_financial_impact
from kpi_rca.reporting.llm import TemplateNarrator, get_narrator, narrate_with_guardrails
from kpi_rca.retrieval.embeddings import make_embedder
from kpi_rca.retrieval.index import TicketIndex
from kpi_rca.retrieval.query import build_query
from kpi_rca.schemas import AnomalyWindow, RCAResult, SegmentFinding

log = logging.getLogger(__name__)

PILOT_DAYS = 3  # segment shares and conversion are stationary, so a short pilot is enough
MIN_GAP_H = 12  # hours between any two injected events
END_MARGIN_H = 6  # no incident ends in the last hours (the replay and ticket search need room after it)
SURGE_EARLIEST_DAY = 2  # batch STL needs a couple of days before a surge
STREAM_PAD_H = 3  # the "incidents" stream mode replays [start - 3 h, end + 3 h)
BR = "country=BR"  # the low-conversion country that must never be blamed

ProgressFn = Callable[[str], None]


@dataclass
class EvaluationResult:
    metrics: dict
    incidents: pd.DataFrame  # one row per injected incident (scenarios.csv)
    summary_markdown: str
    paths: dict[str, Path] = field(default_factory=dict)


# ------------------------------------------------------------ segment catalogue


def segment_catalogue(cfg: AppConfig) -> tuple[pd.DataFrame, float]:
    """Eligible incident segments from an incident-free pilot run, and the global conversion.

    Columns: segment (dict attribute -> value), order, share, conversion.
    """
    ev = cfg.evaluation
    pilot_cfg = cfg.data.model_copy(
        update={"days": PILOT_DAYS, "attempts_per_hour": ev.attempts_per_hour, "incidents": [], "benign_events": []}
    )
    tx = generate_scenario(pilot_cfg, ev.seed).transactions
    total = len(tx)
    attributes = list(cfg.rca.attributes)

    counts: dict[tuple, int] = {}  # sorted ((attribute, value), ...) -> attempts
    rows = []
    for order in range(1, ev.max_order + 1):
        for attrs in itertools.combinations(attributes, order):
            grouped = tx.groupby(list(attrs), observed=True)["success"].agg(["size", "mean"])
            for values, row in grouped.iterrows():
                values = values if isinstance(values, tuple) else (values,)
                key = tuple(zip(attrs, map(str, values)))
                counts[key] = int(row["size"])
                rows.append({"key": key, "order": order, "n": int(row["size"]), "conversion": float(row["mean"])})

    eligible = []
    for r in rows:
        share = r["n"] / total
        if not (ev.min_segment_share <= share <= ev.max_segment_share):
            continue
        if any(value == WEB_APP_VERSION for _, value in r["key"]):
            continue  # app_version=web is exactly device_type=desktop: two right answers
        if _is_redundant(r["key"], counts):
            continue
        eligible.append({"segment": dict(r["key"]), "order": r["order"], "share": share, "conversion": r["conversion"]})
    if not eligible:
        raise ValueError("no incident segment satisfies evaluation.min_segment_share / max_segment_share")
    return pd.DataFrame(eligible), float(tx["success"].mean())


def _is_redundant(key: tuple, counts: dict[tuple, int]) -> bool:
    """True when dropping one attribute keeps (almost) the same rows: that attribute adds nothing."""
    if len(key) < 2:
        return False
    for i in range(len(key)):
        parent = key[:i] + key[i + 1 :]
        if counts[key] >= 0.95 * counts[parent]:
            return True
    return False


# ------------------------------------------------------------ scenario sampling


def _theme_for(segment: dict[str, str]) -> tuple[str, str]:
    """(error_code, ticket theme) a real incident in this segment would plausibly produce."""
    gateway = segment.get("payment_gateway")
    if gateway == "PayPal":
        return "PAYPAL_SDK_TIMEOUT", "paypal_timeout"
    if gateway is not None:
        return "GATEWAY_TIMEOUT", "gateway_timeout"
    if "app_version" in segment:
        return "APP_CRASH_ON_PAY", "app_crash_pay"
    if "os" in segment or "device_type" in segment:
        return "CARD_TOKENIZATION_ERROR", "card_tokenization"
    return "THREE_DS_AUTH_FAILED", "three_ds_failure"


def _place(rng: np.random.Generator, durations: list[int], lo_h: int, hi_h: int, taken: list[tuple[int, int]]) -> list[int]:
    """Start hours in [lo_h, hi_h - duration] at least MIN_GAP_H from each other and from `taken`."""
    for _ in range(1000):
        starts, spans = [], list(taken)
        for d in durations:
            if hi_h - d < lo_h:
                break
            s = int(rng.integers(lo_h, hi_h - d + 1))
            if any(s < e + MIN_GAP_H and s + d + MIN_GAP_H > b for b, e in spans):
                break
            starts.append(s)
            spans.append((s, s + d))
        if len(starts) == len(durations):
            return starts
    raise ValueError("could not place the incidents and the surge; use more evaluation.days or shorter events")


def sample_scenario(
    cfg: AppConfig, catalogue: pd.DataFrame, global_conversion: float, rng: np.random.Generator, *, n_incidents: int
) -> tuple[DataConfig, list[dict]]:
    """A DataConfig with random incidents and one surge, plus per-incident facts the generator does not record."""
    ev = cfg.evaluation
    start = cfg.data.start
    hours = ev.days * 24
    earliest = int(np.ceil(24 * max(cfg.rca.baseline_days, cfg.detection.stream_warmup_days)))

    durations = [int(rng.integers(ev.duration_hours[0], ev.duration_hours[1] + 1)) for _ in range(n_incidents)]
    starts = _place(rng, durations, earliest, hours - END_MARGIN_H, [])

    # visibility: hourly binomial sd of the global conversion rate
    sigma = np.sqrt(global_conversion * (1 - global_conversion) / ev.attempts_per_hour)
    incidents, facts = [], []
    for i, (s, d) in enumerate(zip(starts, durations)):
        target = float(rng.uniform(*ev.visibility_sigma))
        # Only segments big enough to reach the target visibility at the highest allowed
        # failure rate; otherwise small segments would be clipped below the visibility range.
        reachable = catalogue["share"] * catalogue["conversion"] * ev.failure_rate_bounds[1] / sigma >= target
        order = int(rng.integers(1, ev.max_order + 1))
        pool = catalogue[reachable & (catalogue["order"] == order)]
        if pool.empty:
            pool = catalogue[reachable] if reachable.any() else catalogue
        seg = pool.iloc[int(rng.integers(len(pool)))]
        # forced failures hit attempts that would have succeeded with probability `conversion`,
        # so global conversion drops by share * conversion * failure_rate
        rate = target * sigma / (seg["share"] * seg["conversion"])
        rate = float(np.clip(rate, *ev.failure_rate_bounds))
        code, theme = _theme_for(seg["segment"])
        incident_id = f"INC-{i + 1:02d}"
        incidents.append(
            IncidentSpec(
                incident_id=incident_id,
                segment=seg["segment"],
                start=start + timedelta(hours=s),
                duration_hours=d,
                failure_rate=rate,
                error_code=code,
                ticket_rate=ev.ticket_rate,
                ticket_themes=[theme],
            )
        )
        facts.append(
            {
                "incident_id": incident_id,
                "order": int(seg["order"]),
                "target_visibility": target,
                # after clipping the failure rate the visibility is not the target any more
                "visibility": rate * seg["share"] * seg["conversion"] / sigma,
            }
        )
    benign = [_sample_surge(cfg, rng, list(zip(starts, [s + d for s, d in zip(starts, durations)])))]
    data_cfg = cfg.data.model_copy(
        update={
            "days": ev.days,
            "attempts_per_hour": ev.attempts_per_hour,
            "incidents": incidents,
            "benign_events": benign,
        }
    )
    return data_cfg, facts


def _sample_surge(cfg: AppConfig, rng: np.random.Generator, taken: list[tuple[int, int]]) -> BenignEventSpec:
    ev = cfg.evaluation
    duration = int(rng.integers(ev.surge_duration_hours[0], ev.surge_duration_hours[1] + 1))
    [s] = _place(rng, [duration], 24 * SURGE_EARLIEST_DAY, ev.days * 24 - END_MARGIN_H, taken)
    countries = list(cfg.data.countries)
    return BenignEventSpec(
        event_id="SURGE-01",
        segment={"country": countries[int(rng.integers(len(countries)))]},
        start=cfg.data.start + timedelta(hours=s),
        duration_hours=duration,
        volume_multiplier=float(rng.uniform(*ev.surge_multiplier)),
        description="random marketing campaign",
    )


# --------------------------------------------------------------- scoring helpers


def _overlaps(w: AnomalyWindow, start: datetime, end: datetime) -> bool:
    return w.start < end and w.end > start


def _items(finding: SegmentFinding) -> dict[str, str]:
    return {a.attribute: a.value for a in finding.attributes}


def _segment_mask(tx: pd.DataFrame, segment: dict[str, str]) -> np.ndarray:
    mask = np.ones(len(tx), dtype=bool)
    for attribute, value in segment.items():
        mask &= (tx[attribute].astype(str) == value).to_numpy()
    return mask


def score_rca(result: RCAResult | None, truth: dict[str, str], window_tx: pd.DataFrame) -> dict:
    """Exact match, set Jaccard, row-level Jaccard and top-3 of one RCA result."""
    if result is None or not result.findings:
        return {"found": False, "exact": False, "top3": False, "jaccard": 0.0, "row_jaccard": 0.0, "br_blamed": False}
    top = _items(result.findings[0])
    truth_items, top_items = set(truth.items()), set(top.items())
    a, b = _segment_mask(window_tx, top), _segment_mask(window_tx, truth)
    union = int((a | b).sum())
    br_is_innocent = truth.get("country") != "BR"  # blaming BR is only wrong when BR is not the answer
    return {
        "found": True,
        "exact": top == truth,
        "top3": any(_items(f) == truth for f in result.findings[:3]),
        "jaccard": len(truth_items & top_items) / len(truth_items | top_items),
        "row_jaccard": int((a & b).sum()) / union if union else 0.0,
        "br_blamed": br_is_innocent and any(_items(f).get("country") == "BR" for f in result.findings),
    }


def br_rank(result: RCAResult | None) -> int | None:
    """1-based rank of country=BR in feature_importance (None: not in the top 20)."""
    if result is None:
        return None
    ranked = sorted(result.feature_importance, key=result.feature_importance.get, reverse=True)
    return ranked.index(BR) + 1 if BR in ranked else None


def retrieval_scores(hit_ids: list[str], relevant: set[str]) -> dict:
    """P@5, P@10, R@10 (with its cap) and R-precision for one ranked list."""
    r = len(relevant)
    hits = [h in relevant for h in hit_ids]
    return {
        "p_at_5": sum(hits[:5]) / 5,
        "p_at_10": sum(hits[:10]) / 10,
        "r_at_10": sum(hits[:10]) / r,
        "r_at_10_cap": min(10, r) / r,  # the best R@10 possible with r relevant tickets
        "r_precision": sum(hits[:r]) / r,
    }


def bootstrap_ci(values: list[float], n_resamples: int, rng: np.random.Generator) -> list[float] | None:
    """95% percentile bootstrap interval of the mean."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if len(x) == 0:
        return None
    means = x[rng.integers(0, len(x), size=(n_resamples, len(x)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _stat(values: list, n_resamples: int, rng: np.random.Generator) -> dict:
    x = [float(v) for v in values if v is not None]
    if not x:
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": float(np.mean(x)), "ci95": bootstrap_ci(x, n_resamples, rng), "n": len(x)}


def _median(values: list) -> float | None:
    x = [v for v in values if v is not None]
    return float(np.median(x)) if x else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or p + r == 0:
        return None if p is None or r is None else 0.0
    return 2 * p * r / (p + r)


# ------------------------------------------------------------ one scenario


@dataclass
class _Embedders:
    """One embedder per backend for the whole run (TicketIndex.build refits LSA on each scenario's tickets)."""

    loaded: dict[str, Any] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: AppConfig) -> _Embedders:
        out = cls()
        for backend in cfg.evaluation.backends:
            try:
                out.loaded[backend] = make_embedder(cfg.retrieval, backend)
            except RuntimeError as exc:  # model not downloadable / package missing
                log.warning("retrieval backend %s skipped: %s", backend, exc)
                # first line only, so the reason fits in one Markdown table cell
                reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                out.skipped[backend] = f"skipped ({reason[:160]})"
        return out


class _RcaClock:
    """Remembers whether an RCA call is the first of the process (it pays the shap import)."""

    first_done = False


def evaluate_scenario(
    cfg: AppConfig, scenario: Scenario, facts: list[dict], embedders: _Embedders, stream_mode: str
) -> tuple[list[dict], dict, list[dict]]:
    """Score one scenario. Returns (incident rows, scenario counts, memo contexts for A4)."""
    truth = scenario.ground_truth
    tx = scenario.transactions
    times = tx["timestamp"].to_numpy()
    bucket_h = bucket_step(cfg.detection.freq) / pd.Timedelta(hours=1)

    t = time.perf_counter()
    kpis = aggregate_kpis(tx, cfg.detection.freq)
    det = detect_anomalies(kpis, cfg.detection)
    detect_s = time.perf_counter() - t
    drops = [w for w in det.windows if w.kpi_type == "kpi_drop"]

    counts = {
        "incidents": len(truth.incidents),
        "detected": 0,
        "drop_windows": len(drops),
        "true_drop_windows": sum(any(_overlaps(w, i.start, i.end) for i in truth.incidents) for w in drops),
        "surges": len(truth.benign_events),
        "surges_called_drop": sum(any(_overlaps(w, b.start, b.end) for w in drops) for b in truth.benign_events),
        "days": (truth.end - truth.start) / timedelta(days=1),
    }
    counts["false_drop_windows"] = counts["drop_windows"] - counts["true_drop_windows"]

    stream_delays = _stream_delays(cfg, kpis, truth.incidents, stream_mode, counts)

    # ticket indexes, one per backend, without the answer key
    tickets = scenario.tickets.drop(columns=[c for c in GROUND_TRUTH_TICKET_COLUMNS if c in scenario.tickets.columns])
    indexes: dict[str, tuple[TicketIndex, float]] = {}
    if not tickets.empty:
        for backend, embedder in embedders.loaded.items():
            t = time.perf_counter()
            indexes[backend] = (TicketIndex(embedder, cfg.retrieval).build(tickets), time.perf_counter() - t)

    rows, contexts = [], []
    facts_by_id = {f["incident_id"]: f for f in facts}
    for inc in truth.incidents:
        row: dict[str, Any] = {
            "incident_id": inc.incident_id,
            "segment": " & ".join(f"{k}={v}" for k, v in inc.segment.items()),
            **facts_by_id.get(inc.incident_id, {}),
            "share": inc.segment_share,
            "failure_rate": inc.failure_rate,
            "duration_h": (inc.end - inc.start) / timedelta(hours=1),
            "tickets": inc.tickets,
            "stream_delay_h": stream_delays.get(inc.incident_id),
        }
        hits = [w for w in drops if _overlaps(w, inc.start, inc.end)]
        row["detected"] = bool(hits)
        if row["stream_delay_h"] is not None:
            row["stream_delay_buckets"] = row["stream_delay_h"] / bucket_h
        rows.append(row)
        if not hits:
            continue
        counts["detected"] += 1
        # the window that overlaps the incident most is the one whose memo would describe it
        window = max(hits, key=lambda w: min(w.end, inc.end) - max(w.start, inc.start))
        row["window_id"] = window.window_id
        lo, hi = np.searchsorted(times, [np.datetime64(window.start), np.datetime64(window.end)])
        window_tx = tx.iloc[lo:hi]

        results = _run_rca(cfg, tx, window, det.windows, row)
        for name, result in results.items():
            for metric, value in score_rca(result, inc.segment, window_tx).items():
                row[f"{name}_{metric}"] = value
        shap = results["shap"]
        row["shap_top"] = shap.findings[0].label() if shap and shap.findings else None
        row["br_rank_offset"] = br_rank(shap)
        row["br_rank_no_offset"] = br_rank(results["no_offset"])

        top = shap.findings[0] if shap and shap.findings else None
        retrieval_for_memo, search_s = None, 0.0
        relevant = set(scenario.tickets.loc[scenario.tickets["gt_incident_id"] == inc.incident_id, "ticket_id"])
        for backend, (index, build_s) in indexes.items():
            if top is None or not relevant:
                continue
            query = build_query(top, list(top.top_error_codes), cfg.retrieval)
            k = max(10, len(relevant))
            t = time.perf_counter()
            boosted = index.search(query, window.start, window.end, finding=top, k=k)
            elapsed = time.perf_counter() - t
            plain = index.search(query, window.start, window.end, finding=None, k=k)  # A5: no boosts
            for tag, res in (("boost", boosted), ("noboost", plain)):
                for metric, value in retrieval_scores([h.ticket_id for h in res.hits], relevant).items():
                    row[f"{backend}_{tag}_{metric}"] = value
            if retrieval_for_memo is None:
                retrieval_for_memo, search_s = boosted, build_s + elapsed
                row["memo_backend"] = backend

        if shap is not None:
            t = time.perf_counter()
            impact = compute_financial_impact(window, det.scores, top, cfg.reporting)
            context = build_incident_context(window, shap, retrieval_for_memo, impact, cfg.reporting)
            _, guard = narrate_with_guardrails(TemplateNarrator(), context, cfg.llm, cfg.reporting)
            narrate_s = time.perf_counter() - t
            row["template_guardrails_passed"] = guard.passed
            if inc.lost_revenue:
                # impact.py against the generator's counterfactual loss (relative error; negative = underestimate)
                row["loss_true_usd"] = inc.lost_revenue
                row["loss_estimated_usd"] = impact.lost_revenue
                row["loss_rel_error"] = (impact.lost_revenue - inc.lost_revenue) / inc.lost_revenue
            contexts.append(context)
            # compute time from data to memo for this incident (batch detection + this window's stages)
            row["diagnosis_compute_s"] = detect_s + row["shap_runtime_s"] + search_s + narrate_s
    return rows, counts, contexts


def _run_rca(cfg: AppConfig, tx: pd.DataFrame, window: AnomalyWindow, windows: list[AnomalyWindow], row: dict) -> dict:
    """Guided search, brute force and guided search without the M0 offset (A2), with their costs."""
    runs = {
        "shap": lambda: analyze_window(tx, window, cfg.rca, exclude_windows=windows),
        "brute": lambda: brute_force_window(tx, window, cfg.rca, exclude_windows=windows),
        "no_offset": lambda: analyze_window(tx, window, cfg.rca, use_offset=False, exclude_windows=windows),
    }
    results: dict[str, RCAResult | None] = {}
    for name, call in runs.items():
        try:
            result = call()
        except ValueError as exc:
            log.warning("rca (%s) failed for %s: %s", name, window.window_id, exc)
            results[name] = None
            row[f"{name}_error"] = str(exc)  # scored as "not found", but kept visible in scenarios.csv
            continue
        results[name] = result
        row[f"{name}_slices"] = result.slices_evaluated
        row[f"{name}_runtime_s"] = result.runtime_s
        if name == "shap":
            row["shap_cold"] = not _RcaClock.first_done
            _RcaClock.first_done = True
    return results


def _stream_delays(cfg: AppConfig, kpis: pd.DataFrame, incidents: list, mode: str, counts: dict) -> dict[str, float | None]:
    """Streaming detection delay per incident (hours); in "full" mode also streaming precision counts."""
    if mode == "off":
        return {}
    detector = StreamingDetector(cfg.detection)
    if mode == "full":
        result = detector.replay(kpis)
        drops = [w for w in result.windows if w.kpi_type == "kpi_drop"]
        counts["stream_drop_windows"] = len(drops)
        counts["stream_true_drop_windows"] = sum(any(_overlaps(w, i.start, i.end) for i in incidents) for w in drops)
        delays = detection_delays(result.windows, incidents)
    else:
        delays = []
        for inc in incidents:
            pad = timedelta(hours=STREAM_PAD_H)
            result = detector.replay(kpis, start=inc.start - pad, end=inc.end + pad)
            delays.extend(detection_delays(result.windows, [inc]))
    counts["stream_detected"] = sum(d.detected for d in delays)
    return {d.incident_id: d.delay_h for d in delays}


# ----------------------------------------------------------------- the run


def run_evaluation(
    cfg: AppConfig,
    out_dir: str | Path,
    *,
    n_scenarios: int | None = None,
    seed: int | None = None,
    stream_mode: str | None = None,
    with_llm: bool = False,
    progress: ProgressFn | None = None,
) -> EvaluationResult:
    """Run the benchmark and write metrics.json, summary.md and scenarios.csv to out_dir/evaluation."""
    ev = cfg.evaluation.model_copy(
        update={
            "n_scenarios": n_scenarios or cfg.evaluation.n_scenarios,
            "seed": cfg.evaluation.seed if seed is None else seed,
            "stream_mode": stream_mode or cfg.evaluation.stream_mode,
        }
    )
    cfg = cfg.model_copy(update={"evaluation": ev})
    say = progress or log.info
    started = time.perf_counter()

    catalogue, global_conversion = segment_catalogue(cfg)
    say(f"segment catalogue: {len(catalogue)} eligible segments; global conversion {global_conversion:.3f}")
    embedders = _Embedders.load(cfg)

    rows, counts, contexts = [], [], []
    for s in range(ev.n_scenarios):
        rng = np.random.default_rng([ev.seed, s])
        n_inc = int(rng.integers(ev.min_incidents, ev.max_incidents + 1))
        data_cfg, facts = sample_scenario(cfg, catalogue, global_conversion, rng, n_incidents=n_inc)
        scenario_seed = ev.seed * 1000 + s
        t = time.perf_counter()
        scenario = generate_scenario(data_cfg, scenario_seed)
        r, c, ctx = evaluate_scenario(cfg, scenario, facts, embedders, ev.stream_mode)
        for row in r:
            row.update(scenario=s, seed=scenario_seed)
        c.update(scenario=s, seed=scenario_seed)
        rows += r
        counts.append(c)
        contexts += ctx
        say(f"scenario {s + 1}/{ev.n_scenarios}: {c['detected']}/{c['incidents']} detected, "
            f"{c['false_drop_windows']} false drop window(s), {time.perf_counter() - t:.0f} s")

    free = []
    for j in range(ev.incident_free_runs):
        rng = np.random.default_rng([ev.seed, 10_000 + j])
        surge = _sample_surge(cfg, rng, [])
        data_cfg = cfg.data.model_copy(
            update={"days": ev.days, "attempts_per_hour": ev.attempts_per_hour, "incidents": [], "benign_events": [surge]}
        )
        scenario = generate_scenario(data_cfg, ev.seed * 1000 + 900 + j)
        det = detect_anomalies(aggregate_kpis(scenario.transactions, cfg.detection.freq), cfg.detection)
        n_drops = sum(w.kpi_type == "kpi_drop" for w in det.windows)
        free.append({"drop_windows": n_drops, "days": ev.days})
        say(f"incident-free run {j + 1}/{ev.incident_free_runs}: {n_drops} kpi_drop window(s) in {ev.days} days")

    a4 = _run_a4(cfg, contexts, with_llm)
    incidents = pd.DataFrame(rows)
    metrics = summarise(cfg, incidents, counts, free, embedders, a4)
    metrics["settings"]["runtime_s"] = round(time.perf_counter() - started, 1)
    summary = render_summary(metrics)
    paths = write_outputs(Path(out_dir) / "evaluation", metrics, incidents, summary)
    return EvaluationResult(metrics=metrics, incidents=incidents, summary_markdown=summary, paths=paths)


def _run_a4(cfg: AppConfig, contexts: list[dict], with_llm: bool) -> dict:
    """Ablation A4, context arm only: an LLM on the pipeline context, scored by the guardrails.

    The LLM is the openai_compatible narrator when llm.provider says so (free with a
    local server), else Claude.
    """
    # Free-testing policy: A4 may call a paid API, so it runs only when explicitly asked for.
    if not with_llm:
        return {"status": "not run (paid API; free-testing policy; opt in with --with-llm)"}
    if cfg.llm.provider == "openai_compatible":
        llm_cfg = cfg.llm
    else:
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        if not has_key:
            return {"status": "skipped (no API key)"}
        llm_cfg = cfg.llm.model_copy(update={"provider": "claude"})
    try:
        narrator = get_narrator(llm_cfg, memo_max_words=cfg.reporting.memo_max_words)
    except (TypeError, ValueError) as exc:
        return {"status": f"skipped (no usable narrator: {exc})"}
    first, final, fallback, latency = [], [], [], []
    for context in contexts[: cfg.evaluation.a4_max_memos]:
        t = time.perf_counter()
        _, guard = narrate_with_guardrails(narrator, context, llm_cfg, cfg.reporting)
        latency.append(time.perf_counter() - t)
        first.append(bool(guard.attempt_violations) and not guard.attempt_violations[0])
        final.append(guard.passed and not guard.fallback_used)  # an LLM memo that passed, not the template
        fallback.append(guard.fallback_used)
    n = len(first)
    return {
        "status": "run (context arm only; the raw-table arm needs its narrator variant)",
        "narrator": narrator.name,
        "model": narrator.model,
        "n_memos": n,
        "first_attempt_pass_rate": sum(first) / n if n else None,
        "final_pass_rate": sum(final) / n if n else None,
        "fallback_rate": sum(fallback) / n if n else None,
        "latency_s": {"first": latency[0] if latency else None, "median": _median(latency)},
    }


# -------------------------------------------------------------- aggregation


def _col(df: pd.DataFrame, name: str, where: pd.Series | None = None) -> list:
    if name not in df.columns:
        return []
    series = df[name] if where is None else df.loc[where, name]
    return [None if pd.isna(v) else v for v in series.tolist()]


def summarise(
    cfg: AppConfig, df: pd.DataFrame, counts: list[dict], free: list[dict], embedders: _Embedders, a4: dict
) -> dict:
    ev = cfg.evaluation
    rng = np.random.default_rng(ev.seed)
    B = ev.bootstrap_resamples
    stat = lambda values: _stat(values, B, rng)  # noqa: E731 - a short local alias keeps the tables readable

    # ---- detection (batch): precision/recall/F1 with a bootstrap over scenarios ----
    totals = pd.DataFrame(counts)

    def prf(t: pd.DataFrame, det: str, drops: str, true_drops: str) -> tuple[float | None, float | None]:
        r = t[det].sum() / t["incidents"].sum() if t["incidents"].sum() else None
        p = t[true_drops].sum() / t[drops].sum() if t[drops].sum() else None
        return p, r

    def prf_ci(det: str, drops: str, true_drops: str) -> dict:
        p, r = prf(totals, det, drops, true_drops)
        boots = []
        for _ in range(B):
            sample = totals.iloc[rng.integers(0, len(totals), size=len(totals))]
            boots.append(prf(sample, det, drops, true_drops))
        out = {"precision": p, "recall": r, "f1": _f1(p, r)}
        for i, name in enumerate(("precision", "recall")):
            vals = [b[i] for b in boots if b[i] is not None]
            out[f"{name}_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None
        f1s = [f for f in (_f1(*b) for b in boots) if f is not None]
        out["f1_ci95"] = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))] if f1s else None
        return out

    batch = prf_ci("detected", "drop_windows", "true_drop_windows")
    batch.update(
        {
            "n_incidents": int(totals["incidents"].sum()),
            "n_drop_windows": int(totals["drop_windows"].sum()),
            "false_drop_windows": int(totals["false_drop_windows"].sum()),
            "false_drop_windows_per_day": float(totals["false_drop_windows"].sum() / totals["days"].sum()),
            "surge_mislabel_rate": float(totals["surges_called_drop"].sum() / max(totals["surges"].sum(), 1)),
        }
    )
    delays = _col(df, "stream_delay_h")
    streaming: dict[str, Any] = {"mode": ev.stream_mode}
    if ev.stream_mode != "off":
        streaming.update(
            {
                "detected_rate": float(totals["stream_detected"].sum() / totals["incidents"].sum()),
                "median_delay_h": _median(delays),
                "median_delay_buckets": _median(_col(df, "stream_delay_buckets")),
                "delay_h": stat(delays),
            }
        )
    if ev.stream_mode == "full":
        streaming.update(prf_ci("stream_detected", "stream_drop_windows", "stream_true_drop_windows"))
    days_free = sum(f["days"] for f in free)
    false_alarms = {
        "runs": len(free),
        "kpi_drop_windows": sum(f["drop_windows"] for f in free),
        "per_day": sum(f["drop_windows"] for f in free) / days_free if days_free else None,
    }

    # recall by visibility bin (three equal-width bins over the configured range)
    lo, hi = ev.visibility_sigma
    edges = np.linspace(lo, hi, 4)
    by_visibility = []
    if "visibility" in df.columns:
        for a, b in zip(edges[:-1], edges[1:]):
            last = b == edges[-1]
            sel = (df["visibility"] >= a) & ((df["visibility"] <= b) if last else (df["visibility"] < b))
            by_visibility.append(
                {"bin": f"[{a:.1f}, {b:.1f}{']' if last else ')'}", "n": int(sel.sum()),
                 "recall": float(df.loc[sel, "detected"].mean()) if sel.any() else None}
            )
        below = df["visibility"] < lo  # failure_rate clipped at its upper bound
        if below.any():
            by_visibility.append({"bin": f"< {lo:.1f} (failure rate clipped)", "n": int(below.sum()),
                                  "recall": float(df.loc[below, "detected"].mean())})

    # ---- RCA, on detected incidents ----
    detected = df["detected"] if "detected" in df.columns else pd.Series(dtype=bool)
    rca: dict[str, Any] = {"n_detected_incidents": int(detected.sum())}
    for name in ("shap", "brute", "no_offset"):
        rca[name] = {
            m: stat(_col(df, f"{name}_{m}", detected))
            for m in ("found", "exact", "top3", "jaccard", "row_jaccard", "br_blamed")
        }
        rca[name]["slices_median"] = _median(_col(df, f"{name}_slices", detected))
        rca[name]["runtime_s_median"] = _median(_col(df, f"{name}_runtime_s", detected))
    ratios_slices = [s / b for s, b in zip(_col(df, "shap_slices", detected), _col(df, "brute_slices", detected))
                     if s is not None and b]
    ratios_runtime = [s / b for s, b in zip(_col(df, "shap_runtime_s", detected), _col(df, "brute_runtime_s", detected))
                      if s is not None and b]
    cold = df["shap_cold"].fillna(False).astype(bool) if "shap_cold" in df.columns else pd.Series(False, index=df.index)
    rca["cost"] = {
        "slices_ratio": stat(ratios_slices),  # guided / brute force, per incident
        "runtime_ratio": stat(ratios_runtime),
        "shap_runtime_cold_s": (_col(df, "shap_runtime_s", cold) or [None])[0],
        "shap_runtime_warm_median_s": _median(_col(df, "shap_runtime_s", detected & ~cold)),
        "brute_runtime_median_s": _median(_col(df, "brute_runtime_s", detected)),
    }
    br_truth = df["segment"].str.contains(BR, regex=False) if "segment" in df.columns else detected
    rca["a2_offset"] = {
        "br_blamed_with_offset": stat(_col(df, "shap_br_blamed", detected & ~br_truth)),
        "br_blamed_without_offset": stat(_col(df, "no_offset_br_blamed", detected & ~br_truth)),
        "br_rank_with_offset_median": _median(_col(df, "br_rank_offset", detected)),
        "br_rank_without_offset_median": _median(_col(df, "br_rank_no_offset", detected)),
        "br_in_top20_with_offset": stat([v is not None for v in _col(df, "br_rank_offset", detected)]),
        "br_in_top20_without_offset": stat([v is not None for v in _col(df, "br_rank_no_offset", detected)]),
    }
    rca["a3_distractors"] = "not run (W5)"

    # ---- retrieval per backend, boosts on (the pipeline) and off (A5) ----
    retrieval: dict[str, Any] = {}
    for backend in ev.backends:
        if backend in embedders.skipped:
            retrieval[backend] = {"status": embedders.skipped[backend]}
            continue
        retrieval[backend] = {"status": "run"}
        for tag, label in (("boost", "boosts_on"), ("noboost", "boosts_off")):
            retrieval[backend][label] = {
                m: stat(_col(df, f"{backend}_{tag}_{m}"))
                for m in ("p_at_5", "p_at_10", "r_at_10", "r_at_10_cap", "r_precision")
            }

    # None = no memo for that incident (not detected, or RCA failed), so it is not counted
    guard = [bool(v) for v in _col(df, "template_guardrails_passed") if v is not None]
    metrics = {
        "settings": {
            "n_scenarios": ev.n_scenarios,
            "seed": ev.seed,
            "days": ev.days,
            "attempts_per_hour": ev.attempts_per_hour,
            "stream_mode": ev.stream_mode,
            "backends": list(ev.backends),
            "bootstrap_resamples": B,
            "incident_free_runs": ev.incident_free_runs,
            "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        },
        "detection": {"batch": batch, "streaming": streaming, "false_alarms_incident_free": false_alarms,
                      "recall_by_visibility": by_visibility},
        "rca": rca,
        "retrieval": retrieval,
        "guardrails": {"template": {"pass_rate": sum(guard) / len(guard) if guard else None, "n_memos": len(guard)}},
        "a4_llm": a4,
        "impact": {
            "loss_rel_error": stat(_col(df, "loss_rel_error")),
            "loss_abs_rel_error_median": _median([abs(v) for v in _col(df, "loss_rel_error") if v is not None]),
            "note": "estimated lost revenue vs the generator's true loss, per detected incident with a memo",
        },
        "end_to_end": {
            "diagnosis_compute_s": stat(_col(df, "diagnosis_compute_s")),
            "note": "batch detection + RCA + index/search + template memo per incident; excludes data loading",
        },
    }
    return metrics


# ----------------------------------------------------------------- outputs


def _fmt(s: dict | float | None, digits: int = 2) -> str:
    if isinstance(s, dict):
        if s.get("mean") is None:
            return "n/a"
        ci = s.get("ci95")
        return f"{s['mean']:.{digits}f}" + (f" ({ci[0]:.{digits}f}-{ci[1]:.{digits}f})" if ci else "")
    return "n/a" if s is None else f"{s:.{digits}f}"


def _met(value: float | None, target: float, higher_is_better: bool = True) -> str:
    if value is None:
        return "n/a"
    return "yes" if (value >= target if higher_is_better else value <= target) else "no"


def render_summary(m: dict) -> str:
    s, det, rca, ret = m["settings"], m["detection"], m["rca"], m["retrieval"]
    b, st = det["batch"], det["streaming"]

    def ci(value, interval):
        return _fmt(value) + (f" ({interval[0]:.2f}-{interval[1]:.2f})" if interval else "")

    lines = [
        "# Evaluation summary",
        "",
        f"{s['n_scenarios']} scenarios, seed {s['seed']}, {s['days']} days at {s['attempts_per_hour']:.0f} attempts/h; "
        f"streaming mode {s['stream_mode']}; {s['bootstrap_resamples']} bootstrap resamples; "
        f"{s.get('runtime_s', 'n/a')} s. Values are mean (95% bootstrap CI).",
        "",
        "| Experiment | Metric | Value | Target | Met? |",
        "|---|---|---|---|---|",
        f"| Detection (batch) | Recall | {ci(b['recall'], b['recall_ci95'])} | >= 0.90 | {_met(b['recall'], 0.9)} |",
        f"| Detection (batch) | Precision | {ci(b['precision'], b['precision_ci95'])} | >= 0.80 | {_met(b['precision'], 0.8)} |",
        f"| Detection (batch) | F1 | {ci(b['f1'], b['f1_ci95'])} | >= 0.85 | {_met(b['f1'], 0.85)} |",
        f"| Detection (batch) | Surges labelled kpi_drop | {_fmt(b['surge_mislabel_rate'])} | 0 | {_met(b['surge_mislabel_rate'], 0.0, False)} |",
        f"| Detection | False kpi_drop windows/day (incident-free) | {_fmt(det['false_alarms_incident_free']['per_day'])} | report | |",
    ]
    if st["mode"] != "off":
        lines.append(
            f"| Detection (streaming) | Median delay (buckets) | {_fmt(st['median_delay_buckets'], 1)} | <= 3 | "
            f"{_met(st['median_delay_buckets'], 3, False)} |"
        )
        lines.append(f"| Detection (streaming) | Incidents detected | {_fmt(st['detected_rate'])} | report | |")
    if st["mode"] == "full":
        lines.append(f"| Detection (streaming) | F1 | {ci(st['f1'], st['f1_ci95'])} | >= 0.85 | {_met(st['f1'], 0.85)} |")
    sh, br, cost = rca["shap"], rca["brute"], rca["cost"]
    lines += [
        f"| RCA guided | Exact match | {_fmt(sh['exact'])} | >= 0.75 | {_met(sh['exact']['mean'], 0.75)} |",
        f"| RCA guided | Top-3 | {_fmt(sh['top3'])} | >= 0.90 | {_met(sh['top3']['mean'], 0.9)} |",
        f"| RCA guided | Row-level Jaccard | {_fmt(sh['row_jaccard'])} | >= 0.85 | {_met(sh['row_jaccard']['mean'], 0.85)} |",
        f"| RCA guided | BR blamed | {_fmt(sh['br_blamed'])} | 0 | {_met(sh['br_blamed']['mean'], 0.0, False)} |",
        f"| RCA brute force | Exact match | {_fmt(br['exact'])} | comparator | |",
        f"| A3 guided vs brute | Slices ratio | {_fmt(cost['slices_ratio'], 3)} | <= 0.10 | {_met(cost['slices_ratio']['mean'], 0.10, False)} |",
        f"| A3 guided vs brute | Runtime ratio | {_fmt(cost['runtime_ratio'])} | report | |",
        f"| A2 no offset | BR blamed | {_fmt(rca['a2_offset']['br_blamed_without_offset'])} | report | |",
    ]
    for backend, r in ret.items():
        if r["status"] != "run":
            lines.append(f"| Retrieval {backend} | P@5 | {r['status']} | | |")
            continue
        target = 0.8 if backend == "sentence_transformers" else 0.6
        on = r["boosts_on"]
        lines.append(f"| Retrieval {backend} | P@5 | {_fmt(on['p_at_5'])} | >= {target:.2f} | {_met(on['p_at_5']['mean'], target)} |")
    g = m["guardrails"]["template"]
    lines.append(f"| Guardrails (template) | Pass rate | {_fmt(g['pass_rate'])} | 1.00 | {_met(g['pass_rate'], 1.0)} |")
    a4 = m["a4_llm"]
    if "first_attempt_pass_rate" in a4:
        lines.append(f"| Guardrails ({a4.get('model', 'Claude')}) | First-attempt pass | {_fmt(a4['first_attempt_pass_rate'])} | >= 0.80 | "
                     f"{_met(a4['first_attempt_pass_rate'], 0.8)} |")
        lines.append(f"| Guardrails ({a4.get('model', 'Claude')}) | Pass after retries | {_fmt(a4['final_pass_rate'])} | report | |")
        lines.append(f"| Guardrails ({a4.get('model', 'Claude')}) | Template fallback | {_fmt(a4['fallback_rate'])} | report | |")
    else:
        lines.append(f"| A4 raw / context | Hallucination rate | {a4['status']} | | |")
    imp = m.get("impact", {})
    if imp.get("loss_abs_rel_error_median") is not None:
        lines.append(f"| Impact | Lost revenue relative error | {_fmt(imp['loss_rel_error'])} | report | |")
        lines.append(f"| Impact | Median absolute relative error | {_fmt(imp['loss_abs_rel_error_median'])} | report | |")
    e2e = m["end_to_end"]["diagnosis_compute_s"]
    lines.append(f"| End-to-end (template) | Compute per incident (s) | {_fmt(e2e, 1)} | <= 60 | {_met(e2e['mean'], 60, False)} |")

    lines += ["", "## RCA methods (detected incidents)", "",
              "| Method | Found | Exact | Top-3 | Jaccard | Row Jaccard | BR blamed | Median slices | Median runtime (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, label in (("shap", "guided (SHAP)"), ("brute", "brute force"), ("no_offset", "guided, no offset (A2)")):
        r = rca[name]
        lines.append(
            f"| {label} | {_fmt(r['found'])} | {_fmt(r['exact'])} | {_fmt(r['top3'])} | {_fmt(r['jaccard'])} | "
            f"{_fmt(r['row_jaccard'])} | {_fmt(r['br_blamed'])} | {_fmt(r['slices_median'], 0)} | {_fmt(r['runtime_s_median'], 1)} |"
        )
    a2 = rca["a2_offset"]
    lines += [
        "",
        f"Guided search runtime: first call of the process {_fmt(cost['shap_runtime_cold_s'], 1)} s "
        f"(includes the shap import), later calls median {_fmt(cost['shap_runtime_warm_median_s'], 1)} s; "
        f"brute force median {_fmt(cost['brute_runtime_median_s'], 1)} s.",
        f"A2: median rank of {BR} in feature_importance {_fmt(a2['br_rank_with_offset_median'], 0)} with the offset, "
        f"{_fmt(a2['br_rank_without_offset_median'], 0)} without; in the top 20: "
        f"{_fmt(a2['br_in_top20_with_offset'])} vs {_fmt(a2['br_in_top20_without_offset'])}.",
        f"A3 distractor attributes: {rca['a3_distractors']}.",
        "",
        "## Retrieval (P@5, P@10, R@10 with its cap, R-precision)",
        "",
        "| Backend | Boosts | P@5 | P@10 | R@10 | R@10 cap | R-precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for backend, r in ret.items():
        if r["status"] != "run":
            lines.append(f"| {backend} | - | {r['status']} | | | | |")
            continue
        for label in ("boosts_on", "boosts_off"):
            x = r[label]
            lines.append(
                f"| {backend} | {label.split('_')[1]} | {_fmt(x['p_at_5'])} | {_fmt(x['p_at_10'])} | "
                f"{_fmt(x['r_at_10'])} | {_fmt(x['r_at_10_cap'])} | {_fmt(x['r_precision'])} |"
            )
    lines += ["", "## Recall by visibility (hourly binomial sigmas of global conversion)", "",
              "| Bin | Incidents | Recall |", "|---|---|---|"]
    for v in det["recall_by_visibility"]:
        lines.append(f"| {v['bin']} | {v['n']} | {_fmt(v['recall'])} |")
    lines += ["", f"A4: {a4['status']}.", ""]
    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_outputs(out_dir: Path, metrics: dict, incidents: pd.DataFrame, summary: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out_dir / "metrics.json",
        "summary": out_dir / "summary.md",
        "scenarios": out_dir / "scenarios.csv",
    }
    paths["metrics"].write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    paths["summary"].write_text(summary, encoding="utf-8")
    first = [c for c in ("scenario", "seed", "incident_id", "segment") if c in incidents.columns]
    incidents[first + [c for c in incidents.columns if c not in first]].to_csv(
        paths["scenarios"], index=False, encoding="utf-8"
    )
    return paths
