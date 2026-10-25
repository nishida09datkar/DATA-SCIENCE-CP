"""Command line: ``python -m kpi_rca <command>``.

    generate  write a synthetic scenario (transactions, tickets, ground truth) to a data directory
    run       the full pipeline on a data directory; prints every memo
    stream    replay the KPIs bucket by bucket; prints alerts and the delay per known incident
    evaluate  the randomised benchmark (PLAN section 5); writes outputs/evaluation/
    demo      detect on real hourly KPIs, print windows and save a plot to outputs/demo/

Every command returns an exit code instead of raising: 0 = success, 2 = a problem with
the input (bad config, missing data directory, Claude requested without credentials).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from kpi_rca.config import AppConfig, ConfigError, load_config

DEFAULT_CONFIG = Path("config/default.yaml")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_DIR = Path("outputs")

EXIT_OK = 0
EXIT_USAGE = 2

log = logging.getLogger("kpi_rca.cli")


def echo(text: str = "") -> None:
    """Print ASCII only, so a Windows console with a legacy code page never raises UnicodeEncodeError."""
    print(text.encode("ascii", errors="replace").decode("ascii"))


def fail(message: str) -> int:
    print(f"error: {message}".encode("ascii", errors="replace").decode("ascii"), file=sys.stderr)
    return EXIT_USAGE


# ------------------------------------------------------------------ arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kpi_rca", description="Autonomous KPI root-cause analysis engine.")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"YAML config (default {DEFAULT_CONFIG})")

    p = commands.add_parser("generate", help="write a synthetic scenario to a data directory")
    add_config(p)
    p.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--seed", type=int, default=None, help="override the config seed")

    p = commands.add_parser("run", help="detect, explain and write a memo per kpi_drop window")
    add_config(p)
    p.add_argument("--data-dir", "--data", dest="data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--narrator",
        choices=["auto", "claude", "template"],
        default=None,
        help="override llm.provider (default from the config: template, free); claude calls the paid API",
    )
    p.add_argument("--quiet", action="store_true", help="print the window table only, not the memos")

    p = commands.add_parser("stream", help="replay the KPIs bucket by bucket and report detection delays")
    add_config(p)
    p.add_argument("--data-dir", "--data", dest="data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--start", type=datetime.fromisoformat, default=None, help="first bucket to score (ISO time)")
    p.add_argument("--end", type=datetime.fromisoformat, default=None, help="stop before this bucket (ISO time)")

    p = commands.add_parser("evaluate", help="run the randomised benchmark")
    add_config(p)
    p.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--scenarios", type=int, default=None, help="number of scenarios (default evaluation.n_scenarios)")
    p.add_argument("--quick", action="store_true", help="use evaluation.quick_scenarios scenarios")
    p.add_argument("--seed", type=int, default=None, help="override evaluation.seed")
    p.add_argument("--stream", choices=["full", "incidents", "off"], default=None, help="override evaluation.stream_mode")
    p.add_argument("--with-llm", action="store_true", help="also run ablation A4 with Claude (costs money)")

    p = commands.add_parser("demo", help="detect on real hourly KPIs, print the windows and save a plot")
    add_config(p)
    p.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=None,
        help="raw eCommerce CSV(s)/directory or one aggregated hourly KPI table; default: the cached "
        "data/processed/ecom_hourly_*.parquet",
    )
    p.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=Path("outputs/demo"))
    p.add_argument("--period", type=int, default=None, help="seasonal period in buckets (default detection.period)")
    p.add_argument("--seed", type=int, default=None, help="seed for the optional injected evaluation")
    p.add_argument(
        "--evaluate-injected",
        action="store_true",
        help="also score precision/recall on seeded anomalies injected into a copy of the real series",
    )
    p.add_argument("--inject-drops", type=int, default=4, help="injected conversion drops (default 4)")
    p.add_argument("--inject-spikes", type=int, default=3, help="injected traffic spikes (default 3)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        return fail(str(exc))

    handlers = {
        "generate": cmd_generate,
        "run": cmd_run,
        "stream": cmd_stream,
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
    }
    return handlers[args.command](args, cfg)


# ------------------------------------------------------------------- commands


def cmd_generate(args: argparse.Namespace, cfg: AppConfig) -> int:
    from kpi_rca.data import generate_scenario, save_scenario

    seed = cfg.seed if args.seed is None else args.seed
    scenario = generate_scenario(cfg.data, seed)
    paths = save_scenario(scenario, args.out_dir)
    echo(f"generated {len(scenario.transactions):,} transactions and {len(scenario.tickets):,} tickets (seed {seed})")
    for name, path in paths.items():
        echo(f"  {name}: {path}")
    return EXIT_OK


def _check_data_dir(data_dir: Path) -> str | None:
    if not data_dir.is_dir():
        return f"data directory not found: {data_dir} (create one with `python -m kpi_rca generate --out-dir {data_dir}`)"
    return None


def cmd_run(args: argparse.Namespace, cfg: AppConfig) -> int:
    from kpi_rca.pipeline import run_pipeline
    from kpi_rca.reporting.llm import get_narrator

    problem = _check_data_dir(args.data_dir)
    if problem:
        return fail(problem)
    if args.narrator is not None:
        cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"provider": args.narrator})})
    no_credentials = (
        "llm.provider is 'claude' but no Anthropic credentials were found. "
        "Set ANTHROPIC_API_KEY, or use --narrator template."
    )
    # The installed SDK only notices missing credentials at the first request, after
    # detection and RCA have already run, so check up front (the same rule "auto" uses).
    has_credentials = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if cfg.llm.provider == "claude" and not has_credentials:
        return fail(no_credentials)
    try:
        narrator = get_narrator(cfg.llm, memo_max_words=cfg.reporting.memo_max_words)
    except TypeError as exc:  # the Anthropic SDK's error for a client without credentials
        return fail(f"{no_credentials} ({exc})")

    try:
        summary = run_pipeline(cfg, args.data_dir, args.out_dir, narrator_override=narrator)
    except ValueError as exc:  # unreadable or invalid data files
        return fail(str(exc))

    echo(f"narrator: {summary.narrator}; {len(summary.windows)} window(s); {summary.timings_s['total_s']:.1f} s")
    for o in summary.windows:
        w = o.window
        segment = f"  top segment: {o.top_segment}" if o.top_segment else ""
        echo(f"  {w.window_id} {w.kpi_type:<12} {w.start:%Y-%m-%d %H:%M} -> {w.end:%Y-%m-%d %H:%M}  {o.status}{segment}")
    if not args.quiet:
        for o in summary.reports:
            echo()
            echo("=" * 78)
            echo(o.markdown or "")
    echo()
    echo(f"run summary: {summary.summary_path}")
    return EXIT_OK


def cmd_stream(args: argparse.Namespace, cfg: AppConfig) -> int:
    from kpi_rca.data import load_scenario
    from kpi_rca.detection import StreamingDetector, aggregate_kpis, detection_delays

    problem = _check_data_dir(args.data_dir)
    if problem:
        return fail(problem)
    try:
        scenario = load_scenario(args.data_dir)
        kpis = aggregate_kpis(scenario.transactions, cfg.detection.freq)
        detector = StreamingDetector(cfg.detection)
    except ValueError as exc:
        return fail(str(exc))

    def on_alert(alert) -> None:
        echo(f"ALERT {alert.window_id} {alert.kpi_type:<12} started {alert.start:%Y-%m-%d %H:%M}, "
             f"confirmed {alert.detected_at:%Y-%m-%d %H:%M}")

    try:
        result = detector.replay(kpis, start=args.start, end=args.end, on_alert=on_alert)
    except ValueError as exc:  # too little history for the warm-up
        return fail(str(exc))
    echo(f"replayed {result.steps} buckets in {result.runtime_s:.1f} s; {len(result.windows)} window(s)")

    truth = scenario.ground_truth
    if truth is None or not truth.incidents:
        echo("no ground truth in this data directory, so no detection delays")
        return EXIT_OK
    for d in detection_delays(result.windows, truth.incidents):
        if d.detected and d.delay_h is not None:
            echo(f"  {d.incident_id}: detected by {d.window_id} at {d.detected_at:%Y-%m-%d %H:%M}, delay {d.delay_h:.1f} h")
        else:
            echo(f"  {d.incident_id}: not detected in the replayed range")
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace, cfg: AppConfig) -> int:
    from kpi_rca.evaluation import run_evaluation

    n = args.scenarios
    if n is None and args.quick:
        n = cfg.evaluation.quick_scenarios
    if n is not None and n < 1:
        return fail("--scenarios must be at least 1")
    result = run_evaluation(
        cfg,
        args.out_dir,
        n_scenarios=n,
        seed=args.seed,
        stream_mode=args.stream,
        with_llm=args.with_llm,
        progress=lambda line: print(line.encode("ascii", errors="replace").decode("ascii"), file=sys.stderr),
    )
    echo(result.summary_markdown)
    echo(f"written: {', '.join(str(p) for p in result.paths.values())}")
    return EXIT_OK


def cmd_demo(args: argparse.Namespace, cfg: AppConfig) -> int:
    from kpi_rca.demo import run_demo

    inputs = args.input
    if inputs is None:
        cached = sorted(Path("data/processed").glob("ecom_hourly_*.parquet"))
        if not cached:
            return fail(
                "no --input given and no cached ecom_hourly_*.parquet in data/processed; pass the raw "
                "Kaggle CSV files or an aggregated hourly KPI table"
            )
        if len(cached) > 1:
            return fail(f"several cached KPI tables found; pass one with --input: {[str(p) for p in cached]}")
        inputs = cached

    try:
        result = run_demo(
            inputs,
            args.out_dir,
            cfg,
            period=args.period,
            evaluate_injected_anomalies=args.evaluate_injected,
            seed=args.seed,
            n_drops=args.inject_drops,
            n_spikes=args.inject_spikes,
        )
    except ValueError as exc:
        return fail(str(exc))

    d = result.descriptive
    echo(
        f"{d['n_buckets']} hourly buckets, {d['first_bucket']} -> {d['last_bucket']}: "
        f"{d['n_windows']} window(s) ({d['n_kpi_drop']} kpi_drop, {d['n_volume_surge']} volume_surge)"
    )
    echo(f"  kpi_drop hours: {d['kpi_drop_hours']:.0f}; volume_surge hours: {d['volume_surge_hours']:.0f}")
    for w in d["windows"]:
        drop = "" if w["drop_pct"] is None else f"  drop {w['drop_pct'] * 100:.0f}%"
        echo(
            f"  {w['window_id']} {w['kpi_type']:<12} {w['start']} -> {w['end']}  "
            f"{w['n_buckets']} bucket(s){drop}"
        )
    echo(f"  note: {d['note']}")

    if result.injected is not None:
        i = result.injected
        echo("")
        echo(f"injected evaluation (seeded, label '{i['label']}', seed {i['seed']}):")
        echo(
            f"  injected: {i['n_injected']['total']} ({i['n_injected']['drop']} drops, "
            f"{i['n_injected']['spike']} spikes); detected windows: {i['true_positives'] + i['false_positives']}"
        )
        echo(
            f"  precision {_fmt_metric(i['precision'])}, recall {_fmt_metric(i['recall'])}, "
            f"F1 {_fmt_metric(i['f1'])} (TP {i['true_positives']}, FP {i['false_positives']}, "
            f"FN {i['false_negatives']})"
        )
        for kind, s in i["by_kind"].items():
            echo(
                f"  {kind:<5} precision {_fmt_metric(s['precision'])}, recall {_fmt_metric(s['recall'])}, "
                f"n {s['n_injected']}"
            )
        echo(f"  note: {i['note']}")

    echo("")
    echo(f"plot: {result.plot_path}")
    echo(f"windows: {result.windows_path}")
    echo(f"summary: {result.summary_path}")
    return EXIT_OK


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
