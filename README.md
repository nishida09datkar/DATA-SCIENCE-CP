# KPI Root-Cause Analysis Engine

When an online shop's checkout conversion drops, this engine finds when it happened, which customer segment is responsible (for example Android users on app 5.4.0), which support tickets describe it, and writes a short incident memo with the lost revenue and next actions.

Every number is computed deterministically. The memo text comes from a free template by default. An LLM narrator is optional (a free local model through Ollama, or the paid Claude API), and its drafts are always checked by guardrails.

- Plan, methods and targets: [docs/PLAN.md](docs/PLAN.md)
- Current progress and evidence: [docs/STATUS.md](docs/STATUS.md)
- Full verification run (every command and number): [docs/VERIFICATION.md](docs/VERIFICATION.md)
- Live demo script: [docs/DEMO.md](docs/DEMO.md)
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

All tests must pass (315 on 2026-10-23, about 3 minutes). They need no network and no API key.

## Run

```powershell
# 1. Generate the default scenario (21 days, about 1.8M checkout attempts, seed 42) into data/
.venv\Scripts\python -m kpi_rca generate --out data

# 2. Detect drops, find the segment, retrieve tickets, write memos into outputs/reports/
.venv\Scripts\python -m kpi_rca run --data data --out outputs

# 3. Replay the KPIs hour by hour and report how fast each incident is detected
.venv\Scripts\python -m kpi_rca stream --data data

# 4. Benchmark on 20 random scenarios; writes outputs/evaluation/ (about 14 minutes)
.venv\Scripts\python -m kpi_rca evaluate --scenarios 20 --out outputs
```

Add `-v` before the command (`python -m kpi_rca -v run ...`) to log progress. Use `evaluate --quick` for a short smoke run.

### Real data: REES46

[REES46](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) is a real e-commerce event log. Download one monthly file into `data/external/rees46/` (git-ignored, for example `2019-Oct.csv.gz`). The mapping is in `src/kpi_rca/data/adapters.py`:
- one session with a cart or purchase event is one checkout attempt;
- a failure is a cart without a purchase;
- the attributes are category, brand and price band.

REES46 has no ticket text, so the memos cite no customers. It has no labelled incidents either; the script injects them into a copy of the data.

```powershell
# Score detection and RCA on the real month with 20 injected incidents; also export clean/ and injected/ data
.venv\Scripts\python scripts\validate_rees46.py --month 2019-Oct --incidents 20 --export data/rees46

# Run the pipeline on the real month, then on the copy with one injected incident
.venv\Scripts\python -m kpi_rca run --config config/rees46.yaml --data data/rees46/clean --out outputs/rees46_run_clean
.venv\Scripts\python -m kpi_rca run --config config/rees46.yaml --data data/rees46/injected --out outputs/rees46_run_injected
```

`generate` and `evaluate` refuse `config/rees46.yaml` (`dataset: external`), because there is no ground truth to generate or score against. The results are in [docs/VERIFICATION.md](docs/VERIFICATION.md), Phase 3.

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
- `--narrator openai_compatible` is also free.
  - It calls any OpenAI-compatible server at `llm.base_url`. The default is a local [Ollama](https://ollama.com) at `http://localhost:11434/v1`, with model `gemma3:1b` (`ollama pull gemma3:1b`).
  - `LLM_BASE_URL` and `LLM_MODEL` override these. A server that needs a key reads it from `LLM_API_KEY`.
  - Groq's free tier: get a key at [console.groq.com/keys](https://console.groq.com/keys), set `$env:GROQ_API_KEY`, and add `--config config/groq.yaml` (model `openai/gpt-oss-120b`). A rate-limit reply (429) is waited out and resent, up to 3 times.
  - OpenRouter's free models: set `$env:OPENROUTER_API_KEY` and add `--config config/openrouter.yaml` (50 requests a day). Only model IDs ending in `:free` are accepted, so nothing is billed.
  - A larger local model for a 6 GB GPU: `ollama pull gemma3:4b`, then `--config config/ollama_4b.yaml`.
  - A draft that fails the guardrails is retried, then replaced by the template memo.
- `--narrator claude` (or `provider: claude`) calls the paid Anthropic API and needs `ANTHROPIC_API_KEY`. The team does not use it.
- `evaluate` runs ablation A4 and the LLM guardrail criteria only with `--with-llm` (for example `--with-llm --narrator openai_compatible`), even when a key is set.
- Keep any key in an environment variable or a git-ignored `.env`, never in a commit.

## Project layout

```text
src/kpi_rca/
  data/        generator.py (synthetic data), io.py (save, load, validate), adapters.py (REES46)
  detection/   kpi.py (hourly KPIs), detector.py (STL + IsolationForest), streaming.py
  rca/         segment_model.py (LightGBM M0/M1 + SHAP), search.py (guided), baseline.py (brute force)
  retrieval/   query.py, embeddings.py (MiniLM or LSA), index.py (FAISS)
  reporting/   impact.py, context.py, models.py, llm.py, guardrails.py, render.py
  evaluation.py, pipeline.py, cli.py, config.py, schemas.py
tests/         one test file per module
scripts/       validate_rees46.py (real data + injected incidents)
config/        default.yaml, rees46.yaml, groq.yaml, openrouter.yaml, ollama_4b.yaml
docs/          PLAN.md, STATUS.md, VERIFICATION.md, DEMO.md, RESEARCH.md, AI_USE_LOG.md
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
