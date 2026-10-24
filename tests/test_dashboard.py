"""The offline dashboard: built from a template run on the small scenario, no network."""

from __future__ import annotations

import json

import pytest

from kpi_rca.cli import main
from kpi_rca.dashboard import build_dashboard
from kpi_rca.dashboard.build import DATA_GLOBAL, parse_scorecard
from kpi_rca.pipeline import run_pipeline

for module in ("faiss", "lightgbm", "shap", "sklearn", "statsmodels"):
    pytest.importorskip(module)

INCIDENT_SEGMENT = {"country": "DE", "os": "iOS", "payment_gateway": "PayPal"}


@pytest.fixture(autouse=True)
def _no_llm_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture(scope="module")
def offline_cfg(small_cfg):
    return small_cfg.model_copy(
        update={
            "retrieval": small_cfg.retrieval.model_copy(update={"backend": "lsa"}),
            "llm": small_cfg.llm.model_copy(update={"provider": "template"}),
        }
    )


@pytest.fixture(scope="module")
def built(offline_cfg, small_data_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("outputs")
    summary = run_pipeline(offline_cfg, small_data_dir, root / "run")
    index = build_dashboard(offline_cfg, small_data_dir, root / "run", root / "dashboard", outputs_root=root)
    return summary, index


def load_data(dash_dir) -> dict:
    text = (dash_dir / "data.js").read_text(encoding="ascii")
    prefix = f"window.{DATA_GLOBAL} = "
    assert text.startswith(prefix) and text.rstrip().endswith(";")
    return json.loads(text[len(prefix):].rstrip().rstrip(";"))


def test_writes_the_page_and_its_data(built):
    _, index = built
    for name in ("index.html", "style.css", "app.js", "data.js", "vendor/echarts.min.js", "vendor/lucide.min.js"):
        assert (index.parent / name).is_file()
    data = load_data(index.parent)
    assert data["run"]["narrator"] == "template" and data["run"]["has_truth"]
    assert len(data["kpi_hourly"]) == 10 * 24


def test_incidents_match_the_reports_and_the_answer_key(built):
    summary, index = built
    data = load_data(index.parent)
    assert [i["incident_id"] for i in data["incidents"]] == [o.incident_id for o in summary.reports]
    [incident] = [i for i in data["incidents"] if i["segment"] == INCIDENT_SEGMENT]
    assert incident["truth"]["segment_match"] and incident["truth"]["incident_id"] == "INC-T1"
    assert incident["guardrails"]["passed"] and incident["tickets"]
    assert [t["incident_id"] for t in data["truth_incidents"]] == ["INC-T1"]


def test_missing_benchmark_and_real_data_are_empty(built):
    data = load_data(built[1].parent)
    assert data["benchmark"] is None and data["narrators"] == [] and data["rees46"] is None


def test_parse_scorecard_splits_values_and_intervals():
    markdown = "\n".join([
        "# Summary",
        "",
        "| Experiment | Metric | Value | Target | Met? |",
        "|---|---|---|---|---|",
        "| E1 | recall | 0.95 (0.90-0.99) | >= 0.90 | yes |",
        "| E2 | bias | -0.02 (-0.05-0.01) | abs <= 0.10 | no |",
        "| E3 | runtime_s | 4.2 | - | - |",
        "",
        "| not | the | table |",
    ])
    rows = parse_scorecard(markdown)
    assert [r["status"] for r in rows] == ["met", "missed", "reported"]
    assert rows[0]["value"] == 0.95 and rows[0]["ci95"] == [0.90, 0.99]
    assert rows[1]["value"] == -0.02 and rows[1]["ci95"] == [-0.05, 0.01]
    assert rows[2]["ci95"] is None
    assert parse_scorecard("no table here") == []


def test_missing_run_summary_is_an_error(offline_cfg, small_data_dir, tmp_path, capsys):
    with pytest.raises(ValueError):
        build_dashboard(offline_cfg, small_data_dir, tmp_path / "no_run", tmp_path / "dash")
    code = main(["dashboard", "--data-dir", str(small_data_dir), "--run", str(tmp_path / "no_run"),
                 "--out-dir", str(tmp_path / "dash")])
    assert code == 2 and capsys.readouterr().err.startswith("error:")
