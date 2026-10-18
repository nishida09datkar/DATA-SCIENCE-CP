"""The benchmark: scenario sampling, scoring helpers, and one tiny seeded run."""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from kpi_rca import evaluation as E
from kpi_rca.config import config_from_dict
from kpi_rca.schemas import RCAResult, SegmentAttribute, SegmentFinding

for module in ("faiss", "lightgbm", "shap", "sklearn", "statsmodels"):
    pytest.importorskip(module)


def tiny_cfg(**evaluation):
    """10 days at 600 attempts/h: a scenario takes a few seconds. Warm-up 5 days (see StreamingDetector)."""
    return config_from_dict(
        {
            "data": {"attempts_per_hour": 600},
            "rca": {"baseline_days": 3},
            "detection": {"stream_warmup_days": 5, "stream_history_days": 7},
            "reporting": {"make_plots": False},
            "evaluation": {
                "days": 10,
                "attempts_per_hour": 600,
                "n_scenarios": 1,
                "backends": ["lsa"],
                "bootstrap_resamples": 100,
                "incident_free_runs": 1,
                **evaluation,
            },
        }
    )


@pytest.fixture(autouse=True)
def _no_llm_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture(scope="module")
def catalogue():
    return E.segment_catalogue(tiny_cfg())


# ------------------------------------------------------------- sampling


def test_catalogue_respects_the_share_bounds_and_drops_redundant_segments(catalogue):
    cat, conversion = catalogue
    ev = tiny_cfg().evaluation
    assert 0.5 < conversion < 1
    assert cat["share"].between(ev.min_segment_share, ev.max_segment_share).all()
    assert cat["order"].between(1, ev.max_order).all()
    labels = [frozenset(s.items()) for s in cat["segment"]]
    assert frozenset({("os", "Windows"), ("device_type", "desktop")}) not in labels  # same rows as os=Windows
    assert not any(("app_version", "web") in s for s in labels)  # same rows as device_type=desktop


def test_is_redundant():
    counts = {(("os", "Windows"),): 100, (("device_type", "desktop"),): 300, (("os", "Windows"), ("device_type", "desktop")): 100}
    assert E._is_redundant((("os", "Windows"), ("device_type", "desktop")), counts)
    counts[(("os", "Windows"), ("device_type", "desktop"))] = 50
    assert not E._is_redundant((("os", "Windows"), ("device_type", "desktop")), counts)


def test_theme_follows_the_segment():
    assert E._theme_for({"country": "DE", "payment_gateway": "PayPal"}) == ("PAYPAL_SDK_TIMEOUT", "paypal_timeout")
    assert E._theme_for({"payment_gateway": "Adyen"})[1] == "gateway_timeout"
    assert E._theme_for({"os": "Android", "app_version": "5.4.0"})[1] == "app_crash_pay"
    assert E._theme_for({"country": "FR"})[1] == "three_ds_failure"


def test_sampled_scenarios_are_seeded_and_separated(catalogue):
    cfg = tiny_cfg()
    cat, conversion = catalogue
    a = E.sample_scenario(cfg, cat, conversion, np.random.default_rng([7, 0]), n_incidents=2)
    b = E.sample_scenario(cfg, cat, conversion, np.random.default_rng([7, 0]), n_incidents=2)
    assert a[0] == b[0] and a[1] == b[1]  # deterministic per seed

    data_cfg, facts = a
    events = sorted([(i.start, i.end) for i in data_cfg.incidents] + [(e.start, e.end) for e in data_cfg.benign_events])
    for (_, end), (start, _) in zip(events, events[1:]):
        assert start - end >= timedelta(hours=E.MIN_GAP_H)
    earliest = cfg.data.start + timedelta(days=max(cfg.rca.baseline_days, cfg.detection.stream_warmup_days))
    assert all(i.start >= earliest for i in data_cfg.incidents)
    lo, hi = cfg.evaluation.failure_rate_bounds
    assert all(lo <= i.failure_rate <= hi for i in data_cfg.incidents)
    assert [f["incident_id"] for f in facts] == [i.incident_id for i in data_cfg.incidents]


# ------------------------------------------------------------- scoring


def _finding(rank: int, **segment: str) -> SegmentFinding:
    return SegmentFinding(
        rank=rank, attributes=[SegmentAttribute(attribute=a, value=v) for a, v in segment.items()],
        ep=0.9, purity=0.5, score=0.6, failure_rate_window=0.8, failure_rate_baseline=0.1, n_window=10,
        n_baseline=100, excess_failures=7.0, p_value=1e-9, shap_contribution={}, top_error_codes={},
    )


def _result(*findings: SegmentFinding, importance: dict | None = None) -> RCAResult:
    return RCAResult(window_id="W-001", findings=list(findings), feature_importance=importance or {},
                     top_interactions=[], slices_evaluated=5, runtime_s=0.1)


def test_score_rca_exact_jaccard_and_br():
    tx = pd.DataFrame({"country": ["DE", "DE", "DE", "BR"], "os": ["iOS", "iOS", "Android", "iOS"]})
    truth = {"country": "DE", "os": "iOS"}
    exact = E.score_rca(_result(_finding(1, country="DE", os="iOS")), truth, tx)
    assert exact["exact"] and exact["top3"] and exact["jaccard"] == 1.0 and exact["row_jaccard"] == 1.0

    broader = E.score_rca(_result(_finding(1, country="DE"), _finding(2, country="BR")), truth, tx)
    assert not broader["exact"] and not broader["top3"]
    assert broader["jaccard"] == 0.5  # {DE} vs {DE, iOS}
    assert broader["row_jaccard"] == pytest.approx(2 / 3)  # rows 0,1 vs rows 0,1,2
    assert broader["br_blamed"]

    assert E.score_rca(_result(), truth, tx)["found"] is False
    assert E.score_rca(None, truth, tx)["row_jaccard"] == 0.0


def test_br_rank():
    assert E.br_rank(_result(importance={"os=iOS": 0.5, "country=BR": 0.2, "country=DE": 0.3})) == 3
    assert E.br_rank(_result(importance={"os=iOS": 0.5})) is None


def test_retrieval_scores():
    s = E.retrieval_scores(["a", "x", "b", "y", "z", "c"], {"a", "b", "c"})
    assert s["p_at_5"] == 2 / 5 and s["p_at_10"] == 3 / 10
    assert s["r_at_10"] == 1.0 and s["r_at_10_cap"] == 1.0
    assert s["r_precision"] == pytest.approx(2 / 3)  # 2 relevant among the top 3


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    lo, hi = E.bootstrap_ci([0, 1, 1, 1, 0, 1, 1, 1], 500, rng)
    assert lo <= 0.75 <= hi
    assert E.bootstrap_ci([], 100, rng) is None
    assert E.bootstrap_ci([1, 1, 1], 100, rng) == [1.0, 1.0]


# ------------------------------------------------------------- one run


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("evaluation")
    # module-scoped, so it runs before the function-scoped autouse fixture: clear the keys here too
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("ANTHROPIC_API_KEY", raising=False)
        mp.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        return E.run_evaluation(tiny_cfg(), out), out


def test_run_writes_the_three_outputs(tiny_run):
    result, out = tiny_run
    folder = out / "evaluation"
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert {"settings", "detection", "rca", "retrieval", "guardrails", "a4_llm", "end_to_end"} <= set(metrics)
    assert metrics["a4_llm"]["status"].startswith("not run (paid API")
    assert metrics["rca"]["a3_distractors"] == "not run (W5)"
    assert metrics["retrieval"]["lsa"]["status"] == "run"
    assert "| Detection (batch) | Recall |" in (folder / "summary.md").read_text(encoding="utf-8")
    csv = pd.read_csv(folder / "scenarios.csv")
    assert len(csv) == len(result.incidents) >= 1
    assert {"scenario", "incident_id", "segment", "detected", "visibility"} <= set(csv.columns)


def test_template_memos_always_pass_the_guardrails(tiny_run):
    result, _ = tiny_run
    rate = result.metrics["guardrails"]["template"]["pass_rate"]
    assert rate is None or rate == 1.0


def test_a4_is_not_run_without_with_llm_even_when_a_key_is_set(monkeypatch):
    def no_paid_calls(*args, **kwargs):
        raise AssertionError("A4 built a Claude narrator without --with-llm")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(E, "get_narrator", no_paid_calls)
    status = E._run_a4(tiny_cfg(), contexts=[{}], with_llm=False)["status"]
    assert status.startswith("not run (paid API; free-testing policy")


def test_unavailable_backend_is_skipped(monkeypatch):
    def unavailable(cfg, backend=None):
        raise RuntimeError("no network")

    monkeypatch.setattr(E, "make_embedder", unavailable)
    embedders = E._Embedders.load(tiny_cfg(backends=["sentence_transformers"]))
    assert embedders.loaded == {} and embedders.skipped["sentence_transformers"].startswith("skipped")


def test_runs_are_deterministic_apart_from_timings(tiny_run, tmp_path):
    first, _ = tiny_run
    again = E.run_evaluation(tiny_cfg(stream_mode="off", incident_free_runs=0), tmp_path)
    cols = ["scenario", "incident_id", "segment", "failure_rate", "visibility", "detected", "shap_top"]
    cols = [c for c in cols if c in first.incidents.columns]
    pd.testing.assert_frame_equal(first.incidents[cols], again.incidents[cols])
    assert again.metrics["detection"]["streaming"] == {"mode": "off"}
