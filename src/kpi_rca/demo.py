"""Real-data demo: load hourly KPIs, detect windows, save a plot.

Inputs may be:

* raw Kaggle eCommerce behaviour CSV file(s) or a directory of them, loaded with the
  chunked :class:`kpi_rca.data.ecom_kaggle.KaggleEcomLoader`; or
* an already aggregated hourly KPI table (CSV or parquet) with an ``hour``/``bucket``
  column (or a DatetimeIndex) and either ``sessions``/``purchases`` or
  ``attempts``/``successes``.

The output directory defaults to ``outputs/demo``, which is git-ignored. The demo
reports only descriptive results on real data. ``evaluate_injected=True`` additionally
scores the detector on controlled, seeded anomalies injected into a copy of the series;
that is the only place where precision/recall are meaningful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from kpi_rca.config import AppConfig, DetectionConfig
from kpi_rca.data.ecom_kaggle import KaggleEcomLoader, build_kpi_frame_for_pipeline
from kpi_rca.detection import detect_anomalies
from kpi_rca.injected_eval import describe_detection, evaluate_injected
from kpi_rca.schemas import DetectionResult

DEFAULT_DEMO_DIR = Path("outputs/demo")


@dataclass
class DemoResult:
    kpis: pd.DataFrame
    detection: DetectionResult
    descriptive: dict
    plot_path: Path
    windows_path: Path
    summary_path: Path
    injected: dict | None = None


def load_hourly_kpis(
    inputs: str | Path | Sequence[str | Path],
    *,
    period: int = 24,
    freq: str = "1h",
    cache_dir: str | Path | None = Path("data/processed"),
) -> pd.DataFrame:
    """Load raw eCommerce CSVs or an aggregated hourly KPI table into pipeline columns."""
    paths = _as_paths(inputs)
    if len(paths) == 1 and paths[0].is_dir():
        raw = True
    else:
        raw = all(path.is_file() and _looks_like_raw_ecom_csv(path) for path in paths)
    if raw:
        dataset = KaggleEcomLoader(paths, cache_dir=cache_dir, freq=freq).load()
        return build_kpi_frame_for_pipeline(dataset.hourly, period=period, freq=freq)
    if len(paths) != 1:
        raise ValueError(
            "several input files were given but not all look like raw eCommerce CSVs; "
            "pass either raw Kaggle CSVs or one aggregated KPI table"
        )
    return _read_kpi_table(paths[0], period=period, freq=freq)


def run_demo(
    inputs: str | Path | Sequence[str | Path],
    out_dir: str | Path = DEFAULT_DEMO_DIR,
    cfg: AppConfig | None = None,
    *,
    period: int | None = None,
    evaluate_injected_anomalies: bool = False,
    seed: int | None = None,
    n_drops: int = 4,
    n_spikes: int = 3,
) -> DemoResult:
    """Run detection on hourly KPIs and write a plot, windows table and JSON summary."""
    cfg = cfg or AppConfig()
    if period is not None:
        detection = DetectionConfig.model_validate(
            {**cfg.detection.model_dump(), "period": int(period), "weekly_period": 7 * int(period)}
        )
        cfg = cfg.model_copy(update={"detection": detection})
    kpis = load_hourly_kpis(inputs, period=cfg.detection.period, freq=cfg.detection.freq)
    detection = detect_anomalies(kpis, cfg.detection)
    descriptive = describe_detection(detection)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    windows_path = out / "windows.csv"
    pd.DataFrame(descriptive["windows"]).to_csv(windows_path, index=False, encoding="utf-8")
    plot_path = out / "detection.png"
    save_detection_plot(
        detection.scores, detection.windows, plot_path, title="KPI detection on real hourly data"
    )

    injected = None
    if evaluate_injected_anomalies:
        injected = evaluate_injected(kpis, cfg, seed=cfg.seed if seed is None else seed, n_drops=n_drops, n_spikes=n_spikes)

    summary_path = out / "demo_summary.json"
    summary_path.write_text(
        json.dumps({"descriptive": descriptive, "injected": injected}, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return DemoResult(
        kpis=kpis,
        detection=detection,
        descriptive=descriptive,
        plot_path=plot_path,
        windows_path=windows_path,
        summary_path=summary_path,
        injected=injected,
    )


def save_detection_plot(
    scores: pd.DataFrame, windows: Sequence, path: str | Path, *, title: str
) -> Path:
    """Observed vs STL-expected KPIs with detected windows shaded; saves a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = scores.index
    specs = [
        ("attempts", "Attempts"),
        ("conversion_rate", "Conversion rate"),
        ("revenue", "Revenue"),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(15, 10), sharex=True)
    for ax, (column, label) in zip(axes, specs):
        ax.plot(index, scores[column], color="#1f77b4", linewidth=0.8, label="observed")
        expected = f"{column}_expected"
        if expected in scores.columns:
            ax.plot(index, scores[expected], color="#555555", linewidth=0.8, linestyle="--", label="STL expected")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)
    for window in windows:
        color = "#d62728" if window.kpi_type == "kpi_drop" else "#ff7f0e"
        for ax in axes:
            ax.axvspan(window.start, window.end, color=color, alpha=0.18, linewidth=0)
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].set_title(title)
    fig.text(
        0.01,
        0.005,
        "Red = kpi_drop, orange = volume_surge. Real data has no labels: windows are "
        "descriptive, and promotional peaks need not be errors.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ internals


def _as_paths(inputs: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    paths = [Path(p) for p in inputs]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise ValueError(f"input path(s) not found: {missing}")
    return paths


def _looks_like_raw_ecom_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    except OSError:
        return False
    return all(column in header for column in ("event_time", "event_type", "user_session"))


def _read_kpi_table(path: Path, *, period: int = 24, freq: str = "1h") -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported KPI table format {path.suffix!r}; use CSV or parquet")

    frame = _set_time_index(frame, path)
    columns = set(frame.columns)
    if {"sessions", "purchases"} <= columns:
        return build_kpi_frame_for_pipeline(frame, period=period, freq=freq)
    if {"attempts", "successes", "conversion_rate", "revenue"} <= columns:
        frame.index.name = "bucket"
        return frame
    raise ValueError(
        f"hourly KPI table {path} needs sessions/purchases or attempts/successes, conversion_rate and revenue; "
        f"got {sorted(columns)}"
    )


def _set_time_index(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("hour", "bucket", "timestamp", "date", "datetime"):
        if column in frame.columns:
            frame.index = pd.to_datetime(frame.pop(column), utc=True).dt.tz_convert(None)
            frame.index.name = "hour"
            return frame
    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        frame.index.name = "hour"
        return frame
    raise ValueError(
        f"KPI table {path} needs a 'hour'/'bucket' time column or a DatetimeIndex"
    )


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
