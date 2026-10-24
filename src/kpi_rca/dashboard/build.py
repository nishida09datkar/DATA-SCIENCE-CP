"""Build the offline demo dashboard from a pipeline run (``python -m kpi_rca dashboard``).

    run_dir/run_summary.json + run_dir/reports/*.json   the windows and incident reports
    data_dir                                            hourly KPIs, recomputed with the same detector
    data_dir/ground_truth.json (optional)               the answer key, shown beside each finding
    outputs/**/evaluation/metrics.json (optional)       benchmark scorecard and narrator comparison
    outputs/rees46/<month>/ (optional)                  real-data validation

The page is static (index.html, style.css, app.js, plus ECharts, Lucide and the fonts in
vendor/) and reads one generated data.js, so it opens from disk with no server, network or install. The pipeline itself never reads
ground truth; only this presentation layer does, and the page labels it as the answer key.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from kpi_rca.config import AppConfig
from kpi_rca.data import load_scenario
from kpi_rca.detection import aggregate_kpis, detect_anomalies

STATIC_DIR = Path(__file__).parent / "static"
DATA_FILE = "data.js"
DATA_GLOBAL = "KPI_RCA_DATA"
SCORECARD_HEADER = "| Experiment | Metric | Value | Target | Met? |"
_VALUE = re.compile(r"^(-?[\d.]+)(?:\s*\((-?[\d.]+)-(-?[\d.]+)\))?$")


def build_dashboard(
    cfg: AppConfig, data_dir: str | Path, run_dir: str | Path, out_dir: str | Path, outputs_root: str | Path | None = None
) -> Path:
    """Copy the static page (with vendor/) and write data.js to ``out_dir``; returns the index.html path.

    Raises ValueError when the run summary or the data directory is missing.
    """
    data_dir, run_dir, out_dir = Path(data_dir), Path(run_dir), Path(out_dir)
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"no run summary at {summary_path} (write one with `python -m kpi_rca run --out {run_dir}`)")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scenario = load_scenario(data_dir, attributes=cfg.rca.attributes)
    truth = scenario.ground_truth.model_dump(mode="json") if scenario.ground_truth else None
    root = Path(outputs_root) if outputs_root is not None else run_dir.parent

    evaluations = find_evaluations(root)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run": {
            "dir": run_dir.as_posix(),
            "data_dir": data_dir.as_posix(),
            "narrator": summary.get("narrator"),
            "model": summary.get("model"),
            "timings_s": summary.get("timings_s", {}),
            "has_truth": truth is not None,
            "max_retries": cfg.llm.max_retries,
        },
        "kpi_hourly": kpi_rows(scenario.transactions, cfg),
        "windows": [_window_row(w) for w in summary["windows"]],
        "incidents": [incident_row(rep, truth) for rep in _reports(run_dir, summary)],
        "truth_incidents": [
            {k: inc[k] for k in ("incident_id", "start", "end", "segment")} for inc in (truth or {}).get("incidents", [])
        ],
        "benchmark": scorecard(evaluations[-1]) if evaluations else None,
        "narrators": narrator_rows(evaluations),
        "rees46": rees46_rows(root),
    }

    shutil.copytree(STATIC_DIR, out_dir, dirs_exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("</", "<\\/")
    (out_dir / DATA_FILE).write_text(f"window.{DATA_GLOBAL} = {text};\n", encoding="utf-8")
    return out_dir / "index.html"


# ---------------------------------------------------------------- run


def kpi_rows(transactions, cfg: AppConfig) -> list[dict[str, Any]]:
    """Hourly conversion against its expected value, with the detector's label per bucket."""
    scores = detect_anomalies(aggregate_kpis(transactions, cfg.detection.freq), cfg.detection).scores
    return [
        {
            "t": ts.strftime("%Y-%m-%dT%H:%M"),
            "conversion": _num(row.conversion_rate),
            "expected": _num(row.conversion_rate_expected),
            "attempts": int(row.attempts),
            "revenue": _num(row.revenue, 2),
            "label": row.label,
        }
        for ts, row in scores.iterrows()
    ]


def _window_row(w: dict) -> dict[str, Any]:
    return {k: w.get(k) for k in ("window_id", "kpi_type", "start", "end", "status", "incident_id", "top_segment")}


def _reports(run_dir: Path, summary: dict) -> list[dict]:
    reports = []
    for w in summary["windows"]:
        if w.get("incident_id"):
            path = run_dir / "reports" / f"{w['incident_id']}.json"
            if path.is_file():
                reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def segment_label(segment: dict[str, str]) -> str:
    return " & ".join(f"{k}={v}" for k, v in segment.items())


def incident_row(rep: dict, truth: dict | None) -> dict[str, Any]:
    w, imp, nar, guard = rep["window"], rep["impact"], rep["narrative"], rep["guardrails"]
    top = rep["findings"][0] if rep["findings"] else None
    ret = rep.get("retrieval") or {}
    segment = {a["attribute"]: a["value"] for a in top["attributes"]} if top else None
    errors = top["top_error_codes"] if top else {}
    matched = _matching_truth(w["start"], w["end"], truth)
    return {
        "incident_id": rep["incident_id"],
        "window_id": w["window_id"],
        "start": w["start"],
        "end": w["end"],
        "headline": nar["headline"],
        "summary": nar["executive_summary"],
        "root_cause": nar["root_cause_explanation"],
        "confidence": nar["confidence"],
        "severity": imp["severity"],
        "lost_revenue": _num(imp["lost_revenue"], 2),
        "lost_orders": _num(imp["lost_orders"], 1),
        "revenue_at_risk_per_hour": _num(imp["revenue_at_risk_per_hour"], 2),
        "duration_hours": imp["duration_hours"],
        "currency": imp["currency"],
        "conversion_observed": _num(imp["conversion_observed"]),
        "conversion_expected": _num(imp["conversion_expected"]),
        "segment": segment,
        "segment_attempts": top["n_window"] if top else None,
        "failure_rate_window": _num(top["failure_rate_window"]) if top else None,
        "failure_rate_baseline": _num(top["failure_rate_baseline"]) if top else None,
        "share_of_excess_failures": _num(top["ep"]) if top else None,
        "error_codes": [{"code": c, "count": n} for c, n in sorted(errors.items(), key=lambda kv: -kv[1])],
        "shap": [{"feature": f, "contribution": _num(v)} for f, v in (top["shap_contribution"] if top else {}).items()],
        "tickets": [
            {
                "rank": h["rank"],
                "ticket_id": h["ticket_id"],
                "created_at": h["created_at"][:16],
                "channel": h.get("channel"),
                "country": h.get("customer_country"),
                "os": h.get("device_os"),
                "similarity": _num(h["similarity"], 3),
                "text": h["text"],
            }
            for h in ret.get("hits", [])
        ],
        "similar_tickets": ret.get("similar_in_window"),
        "ticket_spike_ratio": _num(ret.get("spike_ratio"), 2),
        "retrieval_backend": ret.get("backend"),
        "actions": nar["recommended_actions"],
        "guardrails": {
            "passed": guard["passed"],
            "attempts": guard["attempts"],
            "narrator_used": guard["narrator_used"],
            "fallback_used": guard["fallback_used"],
            "fallback_reason": guard.get("fallback_reason"),
            "attempt_violations": guard.get("attempt_violations", []),
        },
        "stage_timings_s": rep["provenance"]["stage_timings_s"],
        "truth": _truth_row(matched, segment) if matched else None,
    }


def _matching_truth(start: str, end: str, truth: dict | None) -> dict | None:
    for inc in (truth or {}).get("incidents", []):
        if inc["start"] < end and start < inc["end"]:
            return inc
    return None


def _truth_row(inc: dict, segment: dict | None) -> dict[str, Any]:
    return {
        "incident_id": inc["incident_id"],
        "segment": inc["segment"],
        "error_code": inc.get("error_code"),
        "lost_revenue": inc.get("lost_revenue"),
        "lost_orders": inc.get("lost_orders"),
        "segment_match": segment == inc["segment"],
    }


# ---------------------------------------------------------------- evaluations


def find_evaluations(root: Path) -> list[Path]:
    """Evaluation directories under ``root`` (``evaluation/`` and ``*/evaluation/``), oldest first."""
    dirs = [p.parent for p in [root / "evaluation" / "metrics.json", *root.glob("*/evaluation/metrics.json")] if p.is_file()]
    return sorted(dirs, key=lambda d: (d / "metrics.json").stat().st_mtime)


def parse_scorecard(markdown: str) -> list[dict[str, Any]]:
    """The results table of an evaluation summary.md as rows; values split into mean and 95% CI."""
    lines = markdown.splitlines()
    if SCORECARD_HEADER not in lines:
        return []
    rows = []
    for line in lines[lines.index(SCORECARD_HEADER) + 2:]:
        if not line.startswith("|"):
            break
        experiment, metric, value, target, met = (c.strip() for c in line.strip().strip("|").split("|"))
        m = _VALUE.match(value)
        rows.append({
            "experiment": experiment,
            "metric": metric,
            "value": float(m.group(1)) if m else None,
            "ci95": [float(m.group(2)), float(m.group(3))] if m and m.group(2) else None,
            "text": value,
            "target": target,
            "status": {"yes": "met", "no": "missed"}.get(met, "reported"),
        })
    return rows


def scorecard(eval_dir: Path) -> dict[str, Any] | None:
    summary = eval_dir / "summary.md"
    if not summary.is_file():
        return None
    settings = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8")).get("settings", {})
    return {"source": eval_dir.as_posix(), "settings": settings, "rows": parse_scorecard(summary.read_text(encoding="utf-8"))}


def narrator_rows(eval_dirs: list[Path]) -> list[dict[str, Any]]:
    """Guardrail pass rates per narrator: the template once, then every LLM run of ablation A4."""
    rows, template = [], None
    for d in eval_dirs:
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        template = m.get("guardrails", {}).get("template") or template
        a4 = m.get("a4_llm", {})
        if "first_attempt_pass_rate" in a4:
            rows.append({
                "narrator": a4.get("model") or a4.get("narrator"),
                "kind": "llm",
                "source": d.as_posix(),
                "memos": a4["n_memos"],
                "first_attempt": a4["first_attempt_pass_rate"],
                "after_retries": a4["final_pass_rate"],
                "fallback": a4["fallback_rate"],
                "median_latency_s": _num(a4.get("latency_s", {}).get("median"), 1),
            })
    if template:
        rows.insert(0, {"narrator": "template", "kind": "template", "source": None, "memos": template["n_memos"],
                        "first_attempt": template["pass_rate"], "after_retries": template["pass_rate"],
                        "fallback": 0.0, "median_latency_s": None})
    return rows


# ---------------------------------------------------------------- real data


def rees46_rows(root: Path) -> dict[str, Any] | None:
    """The newest REES46 validation month written by scripts/validate_rees46.py, if any."""
    months = sorted(p.parent for p in root.glob("rees46/*/metrics.json") if (p.parent / "incidents.csv").is_file())
    if not months:
        return None
    month = months[-1]
    m = json.loads((month / "metrics.json").read_text(encoding="utf-8"))
    inc = m["incidents"]
    with open(month / "incidents.csv", encoding="utf-8", newline="") as f:
        rows = [
            {
                "incident": r["incident"],
                "segment": r["segment"],
                "start": r["start"][:16],
                "duration_h": _num(r["duration_h"], 1),
                "failure_rate": _num(r["failure_rate"], 3),
                "visibility_z": _num(r["min_conversion_z"], 2),
                "detected": r["detected"] == "True",
                "found_segment": r.get("shap_top") or None,
                "exact": (r.get("shap_exact") == "True") if r["detected"] == "True" else None,
            }
            for r in csv.DictReader(f)
        ]
    return {
        "month": m["month"],
        "attempts": m["attempts"],
        "conversion": _num(m["conversion"]),
        "days": round(m["days"]),
        "false_drop_windows_per_day": _num(m["clean_stl"]["per_day"], 3),
        "injected": inc["n"],
        "detection_recall": _num(inc["detection_recall"]),
        "rca_exact_guided": _num(inc["shap_exact"]),
        "rca_exact_brute": _num(inc["brute_exact"]),
        "incidents": rows,
    }


def _num(x: Any, digits: int = 4) -> float | None:
    """A rounded float, or None for missing and NaN (JSON has no NaN)."""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else round(value, digits)
