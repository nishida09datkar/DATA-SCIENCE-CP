# KPI Root-Cause Analysis Engine

When an online shop's checkout conversion drops, this engine finds when it happened, which customer segment is responsible (for example Android users on app 5.4.0), which support tickets describe it, and writes a short incident memo with the lost revenue and next actions.

Every number is computed deterministically. The memo text comes from a free template by default; an LLM narrator is optional, paid, and always checked by guardrails.

- Plan, methods and targets: [docs/PLAN.md](docs/PLAN.md)
- Current progress and evidence: [docs/STATUS.md](docs/STATUS.md)
- References: [docs/RESEARCH.md](docs/RESEARCH.md)
- AI-use log: [docs/AI_USE_LOG.md](docs/AI_USE_LOG.md)

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Commands are for Windows PowerShell; on macOS/Linux use `.venv/bin/python` instead of `.venv\Scripts\python`.

```powershell
git clone https://github.com/nishida09datkar/DATA-SCIENCE-CP.git
cd DATA-SCIENCE-CP
uv sync --extra dev --extra embeddings
```

- `--extra dev` installs pytest.
- `--extra embeddings` installs sentence-transformers (and torch) for the MiniLM retrieval backend. It is large; without it, retrieval falls back to the LSA backend with a warning and everything still runs.
- The uv-created venv has no pip. Add packages with `uv add`, not `pip install`.

Check the install:

```powershell
.venv\Scripts\python -m pytest
```

All tests must pass. They need no network and no API key.

## Run

```powershell
# 1. Generate the default scenario (21 days, about 1.8M checkout attempts, seed 42) into data/
.venv\Scripts\python -m kpi_rca generate --out data

# 2. Detect drops, find the segment, retrieve tickets, write memos into outputs/reports/
.venv\Scripts\python -m kpi_rca run --data data --out outputs

# 3. Replay the KPIs hour by hour and report how fast each incident is detected
.venv\Scripts\python -m kpi_rca stream --data data

# 4. Benchmark on 20 random scenarios; writes outputs/evaluation/ (about 11 minutes)
.venv\Scripts\python -m kpi_rca evaluate --scenarios 20 --out outputs
```

Add `-v` before the command (`python -m kpi_rca -v run ...`) to log progress. Use `evaluate --quick` for a short smoke run.

### Outputs

| Path | Written by | Contents |
|---|---|---|
| `data/transactions.parquet`, `tickets.parquet`, `ground_truth.json` | `generate` | Synthetic checkout attempts, tickets, and the injected incidents |
| `outputs/reports/<incident>.md` | `run` | The memo plus the numbers behind it |
| `outputs/reports/<incident>.json` | `run` | Full report with provenance (narrator, attempts, guardrail violations) |
| `outputs/reports/<incident>_kpi.png`, `_shap.png` | `run` | KPI vs expected, and SHAP contributions |
| `outputs/evaluation/metrics.json` | `evaluate` | Every metric with 95% bootstrap CIs |
| `outputs/evaluation/summary.md` | `evaluate` | Results table against the PLAN §5.4 targets |
| `outputs/evaluation/scenarios.csv` | `evaluate` | One row per injected incident |

`data/` and `outputs/` are git-ignored; regenerate them with the commands above.

## Configuration

All settings live in [config/default.yaml](config/default.yaml). Pass another file with `--config my.yaml`; it is deep-merged over the defaults, and unknown keys are rejected with their path.

## Cost: free by default

- The memo narrator is the free template (`llm.provider: template`).
- `--narrator claude` (or `provider: claude`) calls the paid Anthropic API and needs `ANTHROPIC_API_KEY`. The team does not use it.
- `evaluate` runs ablation A4 only with `--with-llm`, even when a key is set.
- Keep any key in an environment variable or a git-ignored `.env`, never in a commit.

## Project layout

```text
src/kpi_rca/
  data/        generator.py (synthetic data), io.py (save, load, validate)
  detection/   kpi.py (hourly KPIs), detector.py (STL + IsolationForest), streaming.py
  rca/         segment_model.py (LightGBM M0/M1 + SHAP), search.py (guided), baseline.py (brute force)
  retrieval/   query.py, embeddings.py (MiniLM or LSA), index.py (FAISS)
  reporting/   impact.py, context.py, models.py, llm.py, guardrails.py, render.py
  evaluation.py, pipeline.py, cli.py, config.py, schemas.py
tests/         one test file per module
config/        default.yaml
docs/          PLAN.md, STATUS.md, RESEARCH.md, AI_USE_LOG.md
```

Stages talk to each other only through the types in `schemas.py` (PLAN §6).

## Contributing

- Branch from `main` as `feat/`, `fix/`, `exp/` or `docs/`; open a PR under about 400 lines that states what, why and how it was tested.
- Merging needs green CI and one approval from another member. Changes to `schemas.py` are labelled "interface" and reviewed by A and the affected owners.
- Run `.venv\Scripts\python -m pytest` before pushing.
- Log every AI-assisted change in [docs/AI_USE_LOG.md](docs/AI_USE_LOG.md).

## Windows notes

- LightGBM needs the Microsoft Visual C++ runtime (it provides OpenMP).
- torch and FAISS/LightGBM may load clashing OpenMP runtimes. Fix the cause (one OpenMP in the environment) rather than setting `KMP_DUPLICATE_LIB_OK`.
- Keep the clone out of OneDrive folders and away from very long paths.
- If the console garbles output, run `chcp 65001` (UTF-8). Plots use matplotlib's Agg backend, so no display is needed.
