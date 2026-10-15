"""The end-to-end pipeline on the small scenario: template narrator, LSA retrieval, no network."""

from __future__ import annotations

import json

import pytest

from kpi_rca import pipeline
from kpi_rca.pipeline import incident_id_for, run_pipeline
from kpi_rca.reporting.llm import TemplateNarrator
from kpi_rca.schemas import RCAResult

for module in ("faiss", "lightgbm", "shap", "sklearn", "statsmodels"):
    pytest.importorskip(module)

INCIDENT = "country=DE & os=iOS & payment_gateway=PayPal"


@pytest.fixture(autouse=True)
def _no_llm_credentials(monkeypatch):
    # provider "auto" would pick Claude when a key is set; these tests never call the API
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
def summary(offline_cfg, small_scenario, tmp_path_factory):
    return run_pipeline(offline_cfg, small_scenario, tmp_path_factory.mktemp("run"))


def test_pipeline_reports_the_incident_with_the_right_segment(summary):
    [report] = [o for o in summary.reports if o.top_segment == INCIDENT]
    assert report.status == "report"
    assert report.incident_id == incident_id_for(report.window)
    assert report.paths["markdown"].is_file() and report.paths["json"].is_file()
    assert "DE" in report.markdown and "PayPal" in report.markdown
    assert summary.narrator == "template"


def test_surge_windows_get_no_memo(summary):
    surges = [o for o in summary.windows if o.window.kpi_type == "volume_surge"]
    assert surges, "the small scenario has a benign surge"
    assert all(o.status == "no memo (volume_surge)" and o.incident_id is None for o in surges)


def test_run_summary_lists_every_window(summary):
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert [w["window_id"] for w in payload["windows"]] == [o.window.window_id for o in summary.windows]
    assert payload["narrator"] == "template" and payload["model"] is None
    assert {"load_s", "detect_s", "index_s", "total_s"} <= set(payload["timings_s"])


def test_report_json_has_provenance(summary):
    report = summary.reports[0]
    data = json.loads(report.paths["json"].read_text(encoding="utf-8"))
    timings = data["provenance"]["stage_timings_s"]
    assert {"rca_s", "retrieval_s", "narrate_s", "detect_s"} <= set(timings)
    assert data["guardrails"]["passed"] is True


def test_ground_truth_never_reaches_the_ticket_index(offline_cfg, small_scenario, tmp_path, monkeypatch):
    seen = []
    real_build = pipeline.TicketIndex.build

    def spy(self, tickets):
        seen.append(set(tickets.columns))
        return real_build(self, tickets)

    monkeypatch.setattr(pipeline.TicketIndex, "build", spy)
    run_pipeline(offline_cfg, small_scenario, tmp_path)
    assert seen and all("gt_incident_id" not in cols for cols in seen)


def test_rca_failure_skips_the_window(offline_cfg, small_scenario, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("window contains no transactions")

    monkeypatch.setattr(pipeline, "analyze_window", broken)
    result = run_pipeline(offline_cfg, small_scenario, tmp_path, narrator_override=TemplateNarrator())
    drops = [o for o in result.windows if o.window.kpi_type == "kpi_drop"]
    assert drops and all(o.status.startswith("skipped: rca failed") for o in drops)
    assert result.reports == []


def test_no_findings_still_writes_a_memo(offline_cfg, small_scenario, tmp_path, monkeypatch):
    def nothing(transactions, window, cfg, use_offset=True, *, exclude_windows=()):
        return RCAResult(window_id=window.window_id, findings=[], feature_importance={}, top_interactions=[],
                         slices_evaluated=0, runtime_s=0.0)

    monkeypatch.setattr(pipeline, "analyze_window", nothing)
    result = run_pipeline(offline_cfg, small_scenario, tmp_path)
    assert result.reports and all(o.top_segment is None for o in result.reports)
    assert all("not isolated" in (o.markdown or "") for o in result.reports)


def test_pipeline_reads_a_data_directory(offline_cfg, small_data_dir, tmp_path, monkeypatch):
    # RCA is replaced so this test only checks loading; the scenario tests above cover RCA
    def nothing(transactions, window, cfg, use_offset=True, *, exclude_windows=()):
        assert "transaction_id" in transactions.columns
        return RCAResult(window_id=window.window_id, findings=[], feature_importance={}, top_interactions=[],
                         slices_evaluated=0, runtime_s=0.0)

    monkeypatch.setattr(pipeline, "analyze_window", nothing)
    result = run_pipeline(offline_cfg, small_data_dir, tmp_path)
    assert result.windows


def test_forced_claude_without_credentials_fails_before_any_work(offline_cfg, small_scenario, tmp_path, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("data was loaded before the narrator check")

    monkeypatch.setattr(pipeline, "load_data", must_not_run)
    cfg = offline_cfg.model_copy(update={"llm": offline_cfg.llm.model_copy(update={"provider": "claude"})})
    with pytest.raises(TypeError, match="ANTHROPIC_API_KEY"):
        run_pipeline(cfg, small_scenario, tmp_path)


def test_missing_data_directory_raises(offline_cfg, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        run_pipeline(offline_cfg, tmp_path / "nope", tmp_path / "out")
