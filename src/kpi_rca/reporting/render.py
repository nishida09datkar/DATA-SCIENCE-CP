"""Writes an IncidentReport to disk: Markdown for people, JSON for machines, PNGs for slides.

Plots use matplotlib's object API (Figure, no pyplot), which renders with the
Agg backend and needs no display, so it works on a server or in CI.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from kpi_rca.reporting.models import IncidentReport

PLOT_CONTEXT = timedelta(hours=48)  # the KPI plot shows this much before and after the window
KPI_TITLES = {"conversion_rate": "Checkout conversion rate", "attempts": "Checkout attempts"}


def render_markdown(report: IncidentReport) -> str:
    memo, impact, window = report.narrative, report.impact, report.window
    cur = impact.currency
    lines = [
        f"# {report.incident_id}: {memo.headline}",
        "",
        f"Window {window.window_id} ({window.kpi_type}): {window.start:%Y-%m-%d %H:%M} to "
        f"{window.end:%Y-%m-%d %H:%M} UTC. Severity {impact.severity}, confidence {memo.confidence}.",
        "",
        "## Executive summary",
        "",
        memo.executive_summary,
        "",
        "## Financial impact",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Lost revenue | {cur} {impact.lost_revenue:,.0f} |",
        f"| Lost orders | {impact.lost_orders:,.0f} |",
        f"| Revenue at risk per hour | {cur} {impact.revenue_at_risk_per_hour:,.0f} |",
        f"| Conversion (observed vs expected) | {impact.conversion_observed:.1%} vs {impact.conversion_expected:.1%} |",
        f"| Duration | {impact.duration_hours:g} h |",
        f"| Severity | {impact.severity} |",
        "",
        "## Root cause",
        "",
        memo.root_cause_explanation,
        "",
    ]
    if report.findings:
        lines += [
            "| Rank | Segment | Failure rate (window) | Failure rate (baseline) | Share of excess failures | Attempts |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {f.rank} | {_cell(f.label())} | {f.failure_rate_window:.1%} | {f.failure_rate_baseline:.1%} "
            f"| {f.ep:.1%} | {f.n_window:,} |"
            for f in report.findings
        ]
    else:
        lines.append("No segment explained the anomaly.")

    lines += ["", "## Customer evidence", ""]
    r = report.retrieval
    if r and r.similar_in_window > 0:
        spike = f", {r.spike_ratio:.1f}x the baseline rate" if r.baseline_rate_per_hour > 0 else ""
        lines.append(f"{r.similar_in_window} similar tickets between {r.search_start:%Y-%m-%d %H:%M} and "
                     f"{r.search_end:%Y-%m-%d %H:%M} UTC{spike} (retrieval backend: {r.backend}).")
    else:
        lines.append("No similar support tickets were found.")
    for quote in memo.customer_evidence:
        lines += ["", f"> {' '.join(quote.split())}"]

    lines += ["", "## Recommended actions", "", "| Priority | Action | Owner | Rationale |", "|---|---|---|---|"]
    lines += [
        f"| {a.priority} | {_cell(a.action)} | {_cell(a.owner_team)} | {_cell(a.rationale)} |"
        for a in memo.recommended_actions
    ]

    g, p = report.guardrails, report.provenance
    status = "passed" if g.passed else "FAILED: " + "; ".join(g.violations)
    timings = ", ".join(f"{stage} {seconds:.2f} s" for stage, seconds in p.stage_timings_s.items())
    lines += [
        "",
        "## Provenance",
        "",
        f"- Narrator: {g.narrator_used} (model: {p.model or 'n/a'})",
        f"- Guardrails: {status} ({g.attempts} narrator attempt(s))",
    ]
    if g.fallback_used:
        lines.append(f"- Fallback reason: {g.fallback_reason}")
    lines += [
        f"- Stage timings: {timings or 'n/a'}",
        f"- Generated at: {p.generated_at:%Y-%m-%d %H:%M} UTC",
        "",
    ]
    return "\n".join(lines)


def save_report(report: IncidentReport, scores: pd.DataFrame, out_dir: Path, make_plots: bool = True) -> dict[str, Path]:
    """Write <id>.md and <id>.json, plus <id>_kpi.png and <id>_shap.png when make_plots is set.

    Returns {"markdown", "json", "kpi_plot", "shap_plot"} -> path; "shap_plot" only with a finding.
    """
    kpi = "attempts" if report.window.kpi_type == "volume_surge" else "conversion_rate"
    if make_plots:  # check before writing anything, so a bad input leaves no half-written report
        for column in (kpi, f"{kpi}_expected"):
            if column not in scores.columns:
                raise ValueError(f"scores is missing column '{column}' needed for the KPI plot")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = report.incident_id
    paths = {"markdown": out_dir / f"{stem}.md", "json": out_dir / f"{stem}.json"}
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["json"].write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if not make_plots:
        return paths

    paths["kpi_plot"] = out_dir / f"{stem}_kpi.png"
    _plot_kpi(report, scores, kpi, paths["kpi_plot"])
    top = report.findings[0] if report.findings else None
    if top and top.shap_contribution:
        paths["shap_plot"] = out_dir / f"{stem}_shap.png"
        _plot_shap(top.shap_contribution, top.label(), paths["shap_plot"])
    return paths


def _plot_kpi(report: IncidentReport, scores: pd.DataFrame, kpi: str, path: Path) -> None:
    w = report.window
    view = scores.loc[w.start - PLOT_CONTEXT : w.end + PLOT_CONTEXT]
    fig = Figure(figsize=(10, 4))
    ax = fig.add_subplot()
    ax.plot(view.index, view[kpi], label="observed", color="tab:blue")
    ax.plot(view.index, view[f"{kpi}_expected"], label="expected (STL)", color="tab:gray", linestyle="--")
    ax.axvspan(w.start, w.end, color="tab:red", alpha=0.15, label=f"anomaly {w.window_id}")
    ax.set_title(f"{report.incident_id}: {KPI_TITLES[kpi]}")
    ax.set_xlabel("hour (UTC)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def _plot_shap(contribution: dict[str, float], segment: str, path: Path) -> None:
    items = sorted(contribution.items(), key=lambda kv: abs(kv[1]))  # largest bar on top
    fig = Figure(figsize=(8, 0.5 * len(items) + 1.5))
    ax = fig.add_subplot()
    ax.barh([k for k, _ in items], [v for _, v in items], color=["tab:red" if v > 0 else "tab:blue" for _, v in items])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Mean SHAP contribution to failure (log-odds)\n{segment}")
    ax.set_xlabel("mean SHAP value over the segment's window rows")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def _cell(text: str) -> str:
    """One Markdown table cell: newlines would end the row and pipes would split it."""
    return " ".join(str(text).split()).replace("|", "\\|")
