"""The end-to-end pipeline (interface spec section 5).

    data -> hourly KPIs -> anomaly windows
    for every kpi_drop window:
        RCA -> ticket retrieval -> financial impact -> incident context -> memo -> report files

volume_surge windows are listed in the run summary but get no memo: more traffic is
not an incident. The ticket index is built once per run, not once per window.

The pipeline never reads ground truth. It works the same on generated data and on
real data that has no ground_truth.json, and the tickets' gt_incident_id column (the
generator's answer key) is dropped before anything sees the tickets.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kpi_rca.config import AppConfig
from kpi_rca.data import Scenario, generate_scenario, load_scenario
from kpi_rca.detection import aggregate_kpis, detect_anomalies
from kpi_rca.rca import analyze_window
from kpi_rca.reporting.context import build_incident_context
from kpi_rca.reporting.impact import compute_financial_impact
from kpi_rca.reporting.llm import Narrator, get_narrator, narrate_with_guardrails
from kpi_rca.reporting.models import IncidentReport, Provenance
from kpi_rca.reporting.render import render_markdown, save_report
from kpi_rca.retrieval.embeddings import make_embedder
from kpi_rca.retrieval.index import TicketIndex
from kpi_rca.retrieval.query import build_query
from kpi_rca.schemas import AnomalyWindow, DetectionResult

log = logging.getLogger(__name__)

GROUND_TRUTH_TICKET_COLUMNS = ("gt_incident_id",)


@dataclass
class WindowOutcome:
    """What the pipeline did with one detected window."""

    window: AnomalyWindow
    status: str  # "report" | "no memo (volume_surge)" | "skipped: <reason>"
    incident_id: str | None = None  # set when a report was written
    top_segment: str | None = None  # label() of the top finding, if any
    paths: dict[str, Path] = field(default_factory=dict)
    markdown: str | None = None  # the rendered memo, for printing


@dataclass
class PipelineSummary:
    out_dir: Path
    narrator: str
    windows: list[WindowOutcome]
    timings_s: dict[str, float]  # shared stages: load, detect, index, total
    summary_path: Path

    @property
    def reports(self) -> list[WindowOutcome]:
        return [w for w in self.windows if w.incident_id is not None]


def incident_id_for(window: AnomalyWindow) -> str:
    return f"INC-{window.start:%Y%m%d-%H%M}"


def load_data(cfg: AppConfig, data: str | Path | Scenario | None) -> Scenario:
    """A Scenario from a data directory, an in-memory Scenario, or (None) a fresh generation."""
    if data is None:
        return generate_scenario(cfg.data, cfg.seed)
    if isinstance(data, Scenario):
        return data
    return load_scenario(data, attributes=cfg.rca.attributes)


def build_ticket_index(tickets: pd.DataFrame | None, cfg: AppConfig) -> TicketIndex | None:
    """One index per run. Returns None when there are no tickets (retrieval is then skipped)."""
    if tickets is None or tickets.empty:
        log.warning("no support tickets; retrieval will be skipped")
        return None
    clean = tickets.drop(columns=[c for c in GROUND_TRUTH_TICKET_COLUMNS if c in tickets.columns])
    return TicketIndex(make_embedder(cfg.retrieval), cfg.retrieval).build(clean)


def run_pipeline(
    cfg: AppConfig,
    data: str | Path | Scenario | None,
    out_dir: str | Path,
    narrator_override: Narrator | None = None,
) -> PipelineSummary:
    """Run the whole pipeline and write reports to ``out_dir/reports``.

    ``narrator_override`` replaces the configured narrator (tests use it). Without it,
    get_narrator picks Claude or the template from cfg.llm; a forced "claude" without
    credentials raises TypeError here, before any work is done.
    """
    started = time.perf_counter()
    out_dir = Path(out_dir)
    narrator = narrator_override or get_narrator(cfg.llm, memo_max_words=cfg.reporting.memo_max_words)
    shared: dict[str, float] = {}

    t = time.perf_counter()
    scenario = load_data(cfg, data)
    transactions = scenario.transactions
    shared["load_s"] = time.perf_counter() - t

    t = time.perf_counter()
    kpis = aggregate_kpis(transactions, cfg.detection.freq)
    det = detect_anomalies(kpis, cfg.detection)
    shared["detect_s"] = time.perf_counter() - t

    t = time.perf_counter()
    drops = [w for w in det.windows if w.kpi_type == "kpi_drop"]
    index = build_ticket_index(scenario.tickets, cfg) if drops else None
    shared["index_s"] = time.perf_counter() - t

    outcomes = []
    for window in det.windows:
        if window.kpi_type != "kpi_drop":
            outcomes.append(WindowOutcome(window=window, status="no memo (volume_surge)"))
            continue
        outcomes.append(_report_window(window, cfg, transactions, det, index, narrator, shared, out_dir))

    shared["total_s"] = time.perf_counter() - started
    summary_path = _write_run_summary(out_dir, outcomes, shared, narrator)
    log.info(
        "pipeline: %d windows, %d reports in %.1fs", len(outcomes), sum(o.incident_id is not None for o in outcomes),
        shared["total_s"],
    )
    return PipelineSummary(
        out_dir=out_dir,
        narrator=narrator.name,
        windows=outcomes,
        timings_s={k: round(v, 3) for k, v in shared.items()},
        summary_path=summary_path,
    )


def _report_window(
    window: AnomalyWindow,
    cfg: AppConfig,
    transactions: pd.DataFrame,
    det: DetectionResult,
    index: TicketIndex | None,
    narrator: Narrator,
    shared: dict[str, float],
    out_dir: Path,
) -> WindowOutcome:
    """RCA -> retrieval -> impact -> context -> narrate -> save, for one kpi_drop window."""
    timings = {k: v for k, v in shared.items()}  # the shared stages count towards every report

    t = time.perf_counter()
    try:
        # Other detected drop windows are kept out of this window's baseline period.
        rca = analyze_window(transactions, window, cfg.rca, exclude_windows=det.windows)
    except ValueError as exc:  # missing columns, or no transactions inside the window
        log.warning("rca failed for %s: %s; window skipped", window.window_id, exc)
        return WindowOutcome(window=window, status=f"skipped: rca failed ({exc})")
    timings["rca_s"] = time.perf_counter() - t

    top = rca.findings[0] if rca.findings else None
    t = time.perf_counter()
    retrieval = None
    if top is not None and index is not None:
        query = build_query(top, list(top.top_error_codes), cfg.retrieval)
        retrieval = index.search(query, window.start, window.end, finding=top)
    timings["retrieval_s"] = time.perf_counter() - t

    t = time.perf_counter()
    impact = compute_financial_impact(window, det.scores, top, cfg.reporting)
    timings["impact_s"] = time.perf_counter() - t

    t = time.perf_counter()
    context = build_incident_context(window, rca, retrieval, impact, cfg.reporting)
    timings["context_s"] = time.perf_counter() - t

    t = time.perf_counter()
    narrative, guard = narrate_with_guardrails(narrator, context, cfg.llm, cfg.reporting)
    timings["narrate_s"] = time.perf_counter() - t

    report = IncidentReport(
        incident_id=incident_id_for(window),
        window=window,
        impact=impact,
        findings=rca.findings,
        retrieval=retrieval,
        narrative=narrative,
        guardrails=guard,
        provenance=Provenance(
            narrator=guard.narrator_used,
            model=narrator.model,
            stage_timings_s={k: round(v, 3) for k, v in timings.items()},
            generated_at=datetime.now(timezone.utc).replace(tzinfo=None),  # naive UTC, as the schema says
        ),
    )
    paths = save_report(report, det.scores, out_dir / "reports", cfg.reporting.make_plots)
    return WindowOutcome(
        window=window,
        status="report",
        incident_id=report.incident_id,
        top_segment=top.label() if top else None,
        paths=paths,
        markdown=render_markdown(report),
    )


def _write_run_summary(
    out_dir: Path, outcomes: list[WindowOutcome], timings: dict[str, float], narrator: Narrator
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "narrator": narrator.name,
        "model": narrator.model,
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "windows": [
            {
                "window_id": o.window.window_id,
                "kpi_type": o.window.kpi_type,
                "start": o.window.start.isoformat(),
                "end": o.window.end.isoformat(),
                "status": o.status,
                "incident_id": o.incident_id,
                "top_segment": o.top_segment,
                "paths": {k: str(p) for k, p in o.paths.items()},
            }
            for o in outcomes
        ],
    }
    path = out_dir / "run_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
