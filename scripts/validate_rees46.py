"""Semi-synthetic validation of detection and RCA on one month of REES46 data (PLAN §3.3, W8).

Real traffic, injected incidents. The mapping onto the transaction schema (one checkout
attempt per user_session; attributes category, brand, price_band) is in
kpi_rca.data.adapters; its docstring lists every column.

1. Clean month: kpi_drop windows per day with STL (the default for under 6 weeks) and with
   MSTL forced on. Nothing is labelled here, so these windows are "unexplained", not
   necessarily false: the month may hold real outages.
2. Injected month, one incident at a time: in one segment (1 or 2 attributes, 3-30% of
   attempts) and a 3-6 h range, turn a share of the successful attempts into failures, on
   a copy of the data. Score detection (a kpi_drop window overlaps the incident) and the
   guided and brute-force RCA of the overlapping window (exact match, top-3, Jaccard), with
   the same scoring as kpi_rca.evaluation.

With --export DIR it also writes two data directories that `python -m kpi_rca run
--config config/rees46.yaml` reads: DIR/clean (the real month, unmodified) and
DIR/injected (a copy with incident R-001 of this seed injected), each with an empty
tickets.parquet (REES46 has no text), plus DIR/injected/injected_incident.json, the label.

Data: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
(Open CDP / REES46). Download a monthly file into data/external/rees46/ (gitignored), e.g.
    curl -sSfL -o data/external/rees46/2019-Oct.csv.gz https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz
Run:
    .venv\\Scripts\\python scripts\\validate_rees46.py --month 2019-Oct --incidents 20 --export data/rees46
Writes outputs/rees46/<month>/metrics.json and incidents.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from kpi_rca.config import DetectionConfig, RCAConfig
from kpi_rca.data import Scenario, save_scenario
from kpi_rca.data.adapters import (
    REES46_ATTRIBUTES,
    candidate_segments,
    inject_incident,
    load_rees46_events,
    no_tickets,
    rees46_transactions,
)
from kpi_rca.detection.detector import detect_anomalies
from kpi_rca.detection.kpi import aggregate_kpis
from kpi_rca.evaluation import _overlaps, score_rca
from kpi_rca.rca.baseline import brute_force_window
from kpi_rca.rca.search import analyze_window
from kpi_rca.rca.segment_model import baseline_failure_probability, fit_baseline_model, prepare_window_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "rees46"
ATTRIBUTES = REES46_ATTRIBUTES


def load_events(month: str) -> pd.DataFrame:
    """Cart and purchase events of one month, cached as parquet next to the raw file."""
    raw = DATA_DIR / f"{month}.csv.gz"
    cache = DATA_DIR / f"{month}.cart_purchase.parquet"
    if not cache.exists() and not raw.exists():
        raise SystemExit(f"{raw} not found; see the download command in this script's docstring")
    return load_rees46_events(raw, cache)


def drop_windows_per_day(tx: pd.DataFrame, det_cfg: DetectionConfig) -> dict:
    kpis = aggregate_kpis(tx, det_cfg.freq)
    windows = [w for w in detect_anomalies(kpis, det_cfg).windows if w.kpi_type == "kpi_drop"]
    days = len(kpis) / 24
    return {
        "drop_windows": len(windows),
        "per_day": len(windows) / days,
        "windows": [f"{w.start:%m-%d %H:%M}..{w.end:%H:%M}" for w in windows],
    }


def draw_incident(rng: np.random.Generator, segments: list[dict[str, str]], tx: pd.DataFrame) -> tuple:
    """(segment, start, end, failure_rate), keeping 7 days of baseline before and 1 day after."""
    t_first, t_last = tx["timestamp"].min(), tx["timestamp"].max()
    segment = segments[rng.integers(len(segments))]
    hours = (t_last - t_first) / timedelta(hours=1)
    start = (t_first + timedelta(hours=int(rng.integers(7 * 24, int(hours) - 24)))).floor("1h")
    end = start + timedelta(hours=int(rng.integers(3, 7)))
    return segment, start, end, float(rng.uniform(0.3, 0.8))


def run(month: str, n_incidents: int, seed: int, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    events = load_events(month)
    tx = rees46_transactions(events)
    load_s = time.perf_counter() - t0
    det_cfg = DetectionConfig()
    rca_cfg = RCAConfig(attributes=ATTRIBUTES)
    days = (tx["timestamp"].max() - tx["timestamp"].min()) / timedelta(days=1)

    metrics: dict = {
        "month": month,
        "events_cart_purchase": len(events),
        "attempts": len(tx),
        "conversion": float(tx["success"].mean()),
        "attempts_per_hour_median": float(tx.groupby(tx["timestamp"].dt.floor("1h")).size().median()),
        "days": days,
        "load_s": load_s,
        "clean_stl": drop_windows_per_day(tx, det_cfg),
        "clean_mstl": drop_windows_per_day(tx, det_cfg.model_copy(update={"mstl_min_weeks": 2.0})),
    }

    rng = np.random.default_rng(seed)
    segments = candidate_segments(tx, ATTRIBUTES)
    rows = []
    times = tx["timestamp"].to_numpy()
    for k in range(n_incidents):
        segment, start, end, failure_rate = draw_incident(rng, segments, tx)
        injected, flipped = inject_incident(tx, segment, start, end, failure_rate, rng)
        in_segment = np.ones(len(tx), dtype=bool)
        for a, v in segment.items():
            in_segment &= (tx[a] == v).to_numpy()
        row = {
            "incident": f"R-{k + 1:03d}",
            "segment": " & ".join(f"{a}={v}" for a, v in segment.items()),
            "order": len(segment),
            "share": float(in_segment.mean()),
            "start": start,
            "duration_h": (end - start) / timedelta(hours=1),
            "failure_rate": failure_rate,
            "flipped": int(flipped.sum()),
        }
        det = detect_anomalies(aggregate_kpis(injected, det_cfg.freq), det_cfg)
        hits = [w for w in det.windows if w.kpi_type == "kpi_drop" and _overlaps(w, start, end)]
        row["detected"] = bool(hits)
        during = det.scores.loc[(det.scores.index >= start) & (det.scores.index < end)]
        row["min_conversion_z"] = float(during["conversion_rate_resid_z"].min())  # why a miss was missed
        if hits:
            window = max(hits, key=lambda w: min(w.end, end) - max(w.start, start))
            row["window"] = f"{window.start:%m-%d %H:%M}..{window.end:%H:%M}"
            lo, hi = np.searchsorted(times, [np.datetime64(window.start), np.datetime64(window.end)])
            window_tx = injected.iloc[lo:hi]
            # RCA's own measure of the window: failures above the M0 baseline (<= 0: no findings)
            data = prepare_window_data(injected, window, rca_cfg, det.windows)
            p0 = baseline_failure_probability(data, fit_baseline_model(data, rca_cfg), rca_cfg)
            row["rca_excess"] = float((data.y_window - p0).sum())
            for name, fn in (("shap", analyze_window), ("brute", brute_force_window)):
                t = time.perf_counter()
                result = fn(injected, window, rca_cfg, exclude_windows=det.windows)
                row[f"{name}_s"] = time.perf_counter() - t
                scores = score_rca(result, segment, window_tx)
                scores.pop("br_blamed")  # synthetic-data check (country=BR); meaningless here
                row.update({f"{name}_{m}": v for m, v in scores.items()})
                row[f"{name}_top"] = result.findings[0].label() if result and result.findings else None
        rows.append(row)
        print(f"{row['incident']} {row['segment']:<40} share={row['share']:.2f} "
              f"fr={failure_rate:.2f} {row['duration_h']:.0f}h minz={row['min_conversion_z']:.1f} detected={row['detected']} "
              f"shap={row.get('shap_top')} brute={row.get('brute_top')}", flush=True)

    frame = pd.DataFrame(rows)
    found = frame[frame["detected"]]

    def mean(col: str) -> float | None:
        return float(found[col].mean()) if col in found and len(found) else None

    metrics["incidents"] = {
        "n": len(frame),
        "detection_recall": float(frame["detected"].mean()),
        **{f"{name}_{m}": mean(f"{name}_{m}") for name in ("shap", "brute")
           for m in ("exact", "top3", "jaccard", "row_jaccard", "s")},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "incidents.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


def export(month: str, seed: int, out: Path) -> None:
    """Write out/clean (real, unmodified) and out/injected (a copy with R-001 of this seed)."""
    tx = rees46_transactions(load_events(month))
    save_scenario(Scenario(transactions=tx, tickets=no_tickets(), ground_truth=None), out / "clean")

    rng = np.random.default_rng(seed)  # the same draws as R-001 in run()
    segment, start, end, failure_rate = draw_incident(rng, candidate_segments(tx, ATTRIBUTES), tx)
    injected, flipped = inject_incident(tx, segment, start, end, failure_rate, rng)
    save_scenario(Scenario(transactions=injected, tickets=no_tickets(), ground_truth=None), out / "injected")
    label = {
        "incident": "R-001",
        "source": f"REES46 {month}, real traffic; this incident is synthetic (injected into a copy)",
        "segment": segment,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "failure_rate": failure_rate,
        "flipped_attempts": int(flipped.sum()),
        "seed": seed,
    }
    (out / "injected" / "injected_incident.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
    print(f"exported {len(tx):,} attempts to {out / 'clean'} and {out / 'injected'}; injected {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", default="2019-Oct")
    parser.add_argument("--incidents", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "rees46")
    parser.add_argument("--export", type=Path, default=None, help="also write clean/ and injected/ data directories")
    args = parser.parse_args()
    if args.export is not None:
        export(args.month, args.seed, args.export)
    if args.incidents < 1:
        return
    metrics = run(args.month, args.incidents, args.seed, args.out / args.month)
    summary = {k: v for k, v in metrics.items() if k not in ("clean_stl", "clean_mstl")}
    summary["clean_stl_per_day"] = metrics["clean_stl"]["per_day"]
    summary["clean_mstl_per_day"] = metrics["clean_mstl"]["per_day"]
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
