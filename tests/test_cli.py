"""The command line: exit codes, messages and one run of each command on small data."""

from __future__ import annotations

import pytest
import yaml

from conftest import small_overrides
from kpi_rca.cli import main

for module in ("faiss", "lightgbm", "shap", "sklearn", "statsmodels"):
    pytest.importorskip(module)


@pytest.fixture(autouse=True)
def _no_llm_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def write_config(path, overrides: dict):
    path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    return path


@pytest.fixture
def small_yaml(tmp_path):
    """The conftest scenario as a YAML file, with offline retrieval and the template narrator."""
    overrides = small_overrides()
    overrides["retrieval"] = {"backend": "lsa"}
    overrides["llm"] = {"provider": "template"}
    return write_config(tmp_path / "small.yaml", overrides)


def test_generate_writes_a_data_directory(tmp_path, capsys):
    overrides = small_overrides()
    overrides["data"]["attempts_per_hour"] = 200  # only the files matter here
    config = write_config(tmp_path / "gen.yaml", overrides)
    assert main(["generate", "--config", str(config), "--out-dir", str(tmp_path / "data"), "--seed", "3"]) == 0
    assert "seed 3" in capsys.readouterr().out
    assert any((tmp_path / "data").iterdir())


def test_missing_data_directory_is_exit_2(small_yaml, tmp_path, capsys):
    assert main(["run", "--config", str(small_yaml), "--data-dir", str(tmp_path / "nope")]) == 2
    err = capsys.readouterr().err
    assert "data directory not found" in err and "generate" in err


def test_bad_config_is_exit_2(tmp_path, capsys):
    config = write_config(tmp_path / "bad.yaml", {"detection": {"no_such_option": 1}})
    assert main(["run", "--config", str(config)]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_claude_without_credentials_is_exit_2(small_yaml, small_data_dir, tmp_path, capsys):
    code = main(["run", "--config", str(small_yaml), "--data-dir", str(small_data_dir),
                 "--out-dir", str(tmp_path), "--narrator", "claude"])
    assert code == 2
    assert "--narrator template" in capsys.readouterr().err


def test_run_prints_the_window_table_and_memos(small_yaml, small_data_dir, tmp_path, capsys):
    code = main(["run", "--config", str(small_yaml), "--data-dir", str(small_data_dir),
                 "--out-dir", str(tmp_path / "out"), "--narrator", "template"])
    out = capsys.readouterr().out
    assert code == 0
    assert "narrator: template" in out and "W-001" in out
    assert "no memo (volume_surge)" in out
    assert "top segment: country=DE & os=iOS & payment_gateway=PayPal" in out
    assert (tmp_path / "out").is_dir()


def test_stream_reports_the_incident_delay(small_yaml, small_data_dir, capsys):
    code = main(["stream", "--config", str(small_yaml), "--data-dir", str(small_data_dir),
                 "--start", "2026-09-08T00:00", "--end", "2026-09-09T00:00"])
    out = capsys.readouterr().out
    assert code == 0
    assert "replayed 24 buckets" in out
    assert "INC-T1" in out


def test_evaluate_with_one_scenario(tmp_path, capsys):
    config = write_config(tmp_path / "eval.yaml", {
        "data": {"attempts_per_hour": 600},
        "rca": {"baseline_days": 3},
        "detection": {"stream_warmup_days": 5, "stream_history_days": 7},
        "reporting": {"make_plots": False},
        "evaluation": {"days": 10, "attempts_per_hour": 600, "backends": ["lsa"],
                       "bootstrap_resamples": 100, "incident_free_runs": 0},
    })
    code = main(["evaluate", "--config", str(config), "--out-dir", str(tmp_path), "--scenarios", "1", "--stream", "off"])
    assert code == 0
    assert "Recall" in capsys.readouterr().out
    assert (tmp_path / "evaluation" / "metrics.json").is_file()


def test_evaluate_rejects_zero_scenarios(tmp_path, capsys):
    assert main(["evaluate", "--out-dir", str(tmp_path), "--scenarios", "0"]) == 2
