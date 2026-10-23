# Verification report (AGENT_PLAN.md, phases 0-4)

Run on 2026-10-23 on Windows 11, Python 3.12 (uv venv), branch `ticket-reporting`. Code commit `c66ed09`; this report is in the commit after it. Every number below comes from a command listed next to it. `data/` and `outputs/` are git-ignored, so the output files are local; the commands regenerate them.

Who did what: an AI agent (Claude Code) ran every command and wrote the new code and this report. Member A has not yet read the new code line by line (see [AI_USE_LOG.md](AI_USE_LOG.md)).

## Phase 0: install and run everything

| Check | Command | Result | Pass |
|---|---|---|---|
| Install | `uv sync --extra dev --extra embeddings` | Installs; MiniLM model downloaded on first use | yes |
| Tests | `.venv\Scripts\python -m pytest` | **315 passed** in 161 s (304 before Phase 3, 280 at the start of the day) | yes |
| Generate | `python -m kpi_rca generate --config config/default.yaml --out data` | 1,765,633 attempts over 21 days; 798 tickets; `ground_truth.json` with INC-001, INC-002, SURGE-US; validation passed | yes |
| Run | `python -m kpi_rca run --data data --out outputs/run_template` | 3 windows in 40.9 s: W-001 kpi_drop, W-002 volume_surge (no memo), W-003 kpi_drop; 2 memos with `.md`, `.json`, 2 PNGs each | yes |
| Stream | `python -m kpi_rca stream --data data` | 336 buckets in 73.1 s; INC-001 and INC-002 detected, delay 2.0 h each; 14 alerts in total, of which 3 kpi_drop alerts match no incident | yes (with the false alerts noted) |
| Evaluate | `python -m kpi_rca evaluate --scenarios 20 --with-llm --narrator openai_compatible --out-dir outputs` | See Phase 1 | yes |

Note: AGENT_PLAN writes `evaluate --out outputs/evaluation`; the command adds `evaluation/` itself, so `--out outputs` is right (otherwise the files land in `outputs/evaluation/evaluation/`).

## Phase 1: the pipeline against its design

### Stage checks on the default scenario (seed 42)

| Stage | Expected | Actual | Pass |
|---|---|---|---|
| Data | 3 files, validated, about 1.8M attempts / 21 days | 1,765,633 attempts, 2026-09-01 to 09-22; `load_scenario` validates on load | yes |
| Detection | INC-001, INC-002 = kpi_drop; SURGE-US = volume_surge, never kpi_drop | W-001 (09-10 13:00-17:00) and W-003 (09-16 17:00-21:00) kpi_drop; W-002 (09-12 18:00-09-13 02:00) volume_surge | yes |
| Streaming | No look-ahead; delay ≤ 3 buckets | 2.0 h for both (1 h buckets); no look-ahead per `test_streaming_never_looks_ahead` | yes |
| RCA | INC-001 = {DE, iOS, PayPal}; INC-002 = {Android, 5.4.0}; BR never blamed | Exactly those two segments as top finding; BR not named | yes |
| Retrieval | Top tickets about the segment, inside the padded window | P@5 = 1.00 for both incidents on both backends; all top-5 inside the padded window (2 h before to 6 h after: 09-10 11:00-23:00 and 09-16 15:00-09-17 03:00; see Phase 2a) | yes |
| Reporting | One memo per incident, grounded, passes all guardrails | 2 memos, guardrails passed on attempt 1 (template) | yes |

### The 19 success criteria (STATUS.md §4)

Source: `.venv\Scripts\python -m kpi_rca evaluate --scenarios 20 --with-llm --narrator openai_compatible --out-dir outputs`. It ran 20 random scenarios (35 incidents, 56 kpi_drop windows) on dev seed 7, in 2,215 s; the log is `outputs/evaluate_log.txt` and the results are `outputs/evaluation/metrics.json` and `summary.md`. Values are mean (95% bootstrap CI).

Every accuracy number equals the 2026-10-15 and earlier 2026-10-23 runs. The 2,215 s (against 848 s before) is mostly the 10 local-LLM memos at about 130 s each. The test suite, the stream replay and the Gate 4 clone ran on the same machine at the same time, which also slows criterion 19.

| # | Criterion | Target | Actual (95% CI) | Met? |
|---|---|---|---|---|
| 1 | Detection recall | ≥ 0.90 | 1.00 (1.00-1.00) | yes |
| 2 | Detection precision | ≥ 0.80 | 0.62 (0.53-0.73); 21 of 56 kpi_drop windows match no incident | **MISS (confirmed)** |
| 3 | Detection F1 | ≥ 0.85 | 0.77 (0.70-0.85) | **MISS (confirmed)** |
| 4 | Surge labelled kpi_drop | 0 | 0.00 | yes |
| 5 | Streaming median delay | ≤ 3 buckets | 2.0 buckets; 0.97 of incidents detected (34/35) | yes |
| 6 | Streaming F1 | ≥ 0.85 | Not measured: needs `detection.stream_mode: full` (the default `incidents` replays only around incidents) | not measured |
| 7 | RCA top-3 | ≥ 0.90 | 0.89 (0.77-0.97), 31 of 35 | **MISS (confirmed)**, by one incident |
| 8 | RCA exact match | ≥ 0.75 | 0.86 (0.74-0.97) | yes |
| 9 | RCA row-level Jaccard | ≥ 0.85 | 0.94 (0.88-0.99) | yes |
| 10 | BR never blamed | 0 | 0.00 | yes |
| 11 | Guided slices vs brute force | ≤ 10% | 2.8% (52 vs 1,892 median slices) | yes |
| 12 | Guided exact within 5 points of brute force | ≤ 5 points | 0.86 vs 0.94 = 8 points | **MISS (confirmed)** |
| 13 | Retrieval P@5, MiniLM | ≥ 0.80 | 0.91 (0.83-0.97) | yes |
| 14 | Retrieval P@5, LSA | ≥ 0.60 | 0.75 (0.69-0.81) | yes |
| 15 | R@10 reported with its cap | report | MiniLM 0.48 (cap 0.54), LSA 0.43 (cap 0.54) | yes |
| 16 | Final memos compliant | 1.00 | 1.00 (template; every memo, including LLM fallbacks, passes) | yes |
| 17 | LLM first-attempt pass | ≥ 0.80 | **0.00** (0/10), gemma3:1b via local Ollama | **MISS** (new; was "not run") |
| 18 | LLM pass after retries | ≥ 0.95 | **0.20** (2/10); 0.80 fell back to the template | **MISS** (new; was "not run") |
| 19 | End-to-end time, template | ≤ 60 s | 14.2 s (13.2-15.2) per incident | yes |

Known misses, as AGENT_PLAN asked:
- **Confirmed:** precision 0.62, F1 0.77, RCA top-3 0.89, guided-vs-brute-force gap 8 points.
- **None refuted.**

Also measured, no target:
- 0.14 false kpi_drop windows per day on the incident-free run.
- Guided RCA is 2.4x *slower* than brute force in wall time (13.2 s vs 5.1 s median).
- Impact against the new loss truth: mean relative error +0.01 (−0.08 to 0.10), median absolute relative error 0.11, over 35 incidents.

Criteria 17-18 need a word of context:
- The model is a 1-billion-parameter local model, the only free LLM option on this laptop.
- The guardrails rejected 8 of 10 memos even after 2 retries; in each case the template memo was used, so no unchecked text reached a memo. Criterion 16 holds.
- A larger free model (for example a free-tier API) may do better. That run needs a team decision and a key (STATUS open issue 4).

## Phase 2: ticket retrieval and reporting (member A)

### 2a. Retrieval

| Check | Evidence | Pass |
|---|---|---|
| MiniLM and LSA both work; `auto` falls back with a warning | `test_auto_falls_back_to_lsa_with_a_warning`; both backends run in `evaluate` | yes |
| FAISS inner product on unit vectors | `tests/test_retrieval.py` index tests | yes |
| Nothing outside the padded window | `test_tickets_outside_the_padded_window_are_never_returned`, and A's own `test_my_retrieval_test_the_window_filter_holds_on_a_generated_scenario` | yes |
| Country is a boost, not query text | `test_query_leaves_out_the_country`; queries below have no country | yes |
| Query words come from config | `test_query_attributes_and_wording_come_from_the_config` (`retrieval.query_attributes`, `retrieval.customer_words`) | yes |
| P@5 targets (MiniLM ≥ 0.80, LSA ≥ 0.60) | Criteria 13-14 above | see table |

Per-incident printout on the default scenario (the top 5 were all labelled relevant on both backends):

| Backend | Incident | Query | Spike ratio | P@5 |
|---|---|---|---|---|
| MiniLM | INC-001 | `PayPal iPhone iOS app payment failed at checkout paypal sdk timeout card declined user abandoned` | 9.2x (39 similar in window) | 1.00 |
| MiniLM | INC-002 | `Android phone app app version 5.4.0 update payment failed at checkout app crash on pay card declined user abandoned` | 14.1x (44) | 1.00 |
| LSA | INC-001 | same | 35.5x (13) | 1.00 |
| LSA | INC-002 | same | 81.9x (6) | 1.00 |

The LSA spike ratios are higher only because its baseline count is near zero (0.03 and 0.01 similar tickets per hour); MiniLM's counts are the more realistic ones.

### 2b. Reporting

| Check | Evidence | Pass |
|---|---|---|
| Impact computed in Python | `tests/test_reporting_impact.py` | yes |
| Context JSON is the only narrator input; ticket text only under `excerpt` | `test_customer_text_appears_only_under_the_untrusted_excerpt_key` | yes |
| `MemoNarrative` schema | `test_invalid_payload_reports_schema_violation` | yes |
| Guardrails: schema, numbers, segment, quotes | `tests/test_guardrails.py`, `tests/test_guardrails_adversarial.py` | yes |
| Readable in 30 s (summary + root cause ≤ 200 words) | `reporting.memo_max_words` is enforced by the guardrail word check | yes |

### 2c. Items built

| # | Item | Where | Evidence | Pass |
|---|---|---|---|---|
| 1 | Adversarial guardrail tests | `tests/test_guardrails_adversarial.py` (18) | Invented revenue, wrong country/OS/gateway, fabricated or paraphrased quote, "$5M" prompt injection in a ticket, too-long headline, too many actions, missing fields: all rejected | yes |
| 2 | A's two tests | `test_my_retrieval_test_the_window_filter_holds_on_a_generated_scenario` (test_retrieval.py), `test_my_reporting_test_an_invented_number_ends_in_the_template_memo` (test_llm.py) | Both pass. **A must still read and defend them** | yes |
| 3 | Free OpenAI-compatible narrator | `reporting/llm.py` `OpenAICompatibleNarrator`; config `llm.provider: openai_compatible`, `base_url` (default local Ollama), `openai_model`, `api_key_env`, `response_format` | JSON-schema output validated against `MemoNarrative`; retries with the guardrail violations as feedback; template fallback. Opt-in only; tests use a fake client, no network | yes |
| 4 | Guardrail metric run with an LLM | Ollama `gemma3:1b` on this laptop (free, local) | Criteria 17-18 above | see table |
| 5 | Lost-revenue truth | `data/generator.py` writes `lost_orders`, `lost_revenue` per incident (the orders that failed only because of the incident); `evaluate` scores `impact.py` against it | `test_data.py` checks the truth against the transactions; `test_reporting_impact.py` checks the estimate against it | yes |
| 6 | `effort` setting | `LLMConfig.effort` (low/medium/high), sent as `reasoning_effort` only when set | `tests/test_llm.py` | yes |

**Gate 2: passed.** The INC-001 memo (in [DEMO.md](DEMO.md)) names DE / iOS / PayPal, PAYPAL_SDK_TIMEOUT, and three verbatim ticket quotes ("PayPal checkout keeps timing out in the iOS app...").

Impact accuracy on the default scenario (estimate from `impact.py` vs the generator's truth):

| Incident | Lost orders est. / true | Lost revenue est. / true |
|---|---|---|
| INC-001 | 589 / 602 (−2%) | USD 33,018 / 38,034 (−13%) |
| INC-002 | 678 / 738 (−8%) | USD 38,635 / 36,360 (+6%) |

Orders are close. Revenue is orders × the window's average order value, so it misses when the failed baskets are larger or smaller than average.

Free LLM run (`run --narrator openai_compatible`, gemma3:1b):
- INC-002 passed on the first attempt.
- INC-001 failed all 3 attempts and fell back to the template, and the JSON says `template_fallback`.
- The violations show why. The model wrapped each quote in extra quotation marks, so they were not verbatim. It also gave 4-5 quotes (max 3) and 5 actions (max 4), and left out DE and iOS.
- So the guardrails did their job. The extra-quotation-mark rejection may be stricter than needed; that is a team decision, not changed.

## Phase 3: own dataset (REES46)

### What is real and what is injected

| Part | Real or not | Detail |
|---|---|---|
| Traffic | **Real** | REES46 / Open CDP, Kaggle `mkechinov/ecommerce-behavior-data-from-multi-category-store`, file `2019-Oct.csv.gz`. 1,669,363 cart and purchase events → 910,796 checkout attempts (one per session), 2019-10-01 to 10-31, conversion 0.691 |
| Outcome | Real, but a different meaning | "Failure" = a cart with no purchase (abandonment), error code `NO_PURCHASE`. REES46 has no payment errors |
| Attributes | Real | `category` (top level of category_code), `brand` (top 20, rest "other"), `price_band` (quartiles). The generator's country / OS / device / gateway / app version do **not** exist in REES46 and were **not** invented |
| Tickets | **None** | REES46 has no text. The ticket table is empty; retrieval finds nothing and the memo says "No similar support tickets were found". No tickets were fabricated |
| Incidents | **Injected**, on a copy | REES46 has no labelled incidents. `inject_incident` fails a share of one segment's successful attempts in one time range, in a copy; the original frame is never changed (`test_injection_changes_a_copy_and_only_the_segment_in_range`). Injected rows look like real failures; the label is kept separately |

Mapping and rules: `src/kpi_rca/data/adapters.py` (module docstring); config: `config/rees46.yaml` (`dataset: external`).

### Commands

```powershell
# raw file into data/external/rees46/ (gitignored), then:
.venv\Scripts\python scripts\validate_rees46.py --month 2019-Oct --incidents 0 --export data/rees46
.venv\Scripts\python -m kpi_rca run --config config/rees46.yaml --data data/rees46/clean    --out outputs/rees46_run_clean
.venv\Scripts\python -m kpi_rca run --config config/rees46.yaml --data data/rees46/injected --out outputs/rees46_run_injected
.venv\Scripts\python scripts\validate_rees46.py --month 2019-Oct --incidents 20
```

`evaluate` refuses `dataset: external` by design: it scores generated scenarios against `ground_truth.json`, which real data does not have (`test_generate_and_evaluate_refuse_an_external_dataset_config`). Instead, `scripts/validate_rees46.py` scores real data plus injected incidents, using the same scoring functions (`score_rca`, `_overlaps`). This replaces "run `evaluate` on the adapted data".

### Results on unmodified real data (`run`, clean)

- 12 windows in 30.2 s. 7 are kpi_drop with memos: 10-01 00:00, 10-03 16:00, 10-06 02:00, 10-13 14:00, 10-15 09:00, 10-21 07:00, 10-24 05:00.
- These match the 7 STL kpi_drop windows of the scoring script (0.23 per day; MSTL forced on: 5, 0.16 per day).
- There are no labels, so these windows are **unexplained, not known false**: the month may hold real outages or logging changes.
- Top segments include brand=samsung (10-03), category=electronics & price_band=p1_low (10-06) and price_band=p4_high (10-15). The 10-01 00:00 window has no baseline before it, so it has no findings.

### Results with one injected incident (`run`, injected)

The injected incident, from `data/rees46/injected/injected_incident.json`:
- brand=apple, 2019-10-25 18:00-23:00;
- failure rate 0.729, 240 attempts flipped;
- true lost revenue USD 221,918 (recomputed by comparing clean vs injected: exactly 240 rows differ).

| Check | Result | Pass |
|---|---|---|
| Detected | W-010 kpi_drop, 10-25 18:00-23:00, the exact injected range | yes |
| RCA | Top segment brand=apple (exact match); 84.6% failure vs 37.8% baseline; holds 78.1% of the excess | yes |
| Memo + guardrails | `outputs/rees46_run_injected/reports/INC-20191025-1800.md`, SEV1, guardrails passed, "No similar support tickets were found" | yes |
| Lost orders | 276 estimated vs 240 true (+15%) | rough |
| Lost revenue | USD 78,383 estimated vs USD 221,918 true (**−65%**) | **MISS** |

The revenue miss: apple baskets are far larger than the window's average order value, which `impact.py` multiplies by. Not fixed (it would mean a segment-level order value; team decision).

Side effect, also a finding: adding these 240 failures on 10-25 changed detection elsewhere in the month. Three clean-run windows (10-03, 10-21, 10-24) disappeared, and 10-06 moved to 03:00. On heavy-tailed real data the Isolation Forest thresholds depend on the whole month, so distant data shifts them.

### Results with 20 injected incidents, one at a time (`validate_rees46.py`, seed 42)

Output: `outputs/rees46/2019-Oct/metrics.json`, `incidents.csv`. The refactor of the script onto `adapters.py` reproduced the earlier run exactly (every accuracy number equal; only timings differ).

| Metric | REES46 (real + injected) | Synthetic (Phase 1) |
|---|---|---|
| Detection recall | **0.30** (6/20); 6/10 with min conversion z ≤ −5, 0/10 above | 1.00 |
| Guided RCA exact (detected) | 0.50 (3/6) | 0.86 |
| Brute-force exact (detected) | 0.67 (4/6) | 0.94 |
| Guided row Jaccard | 0.66 | 0.94 |
| kpi_drop windows per day with no incident | 0.23 (STL), 0.16 (MSTL), unlabelled | 0.14 (incident-free run) |
| Tickets / retrieval | none (no text in REES46) | P@5 0.91 MiniLM |

The 2 detected incidents with no findings (R-002, R-015) had negative excess over the 7-day M0 baseline (−103, −192): heavier cart logging in the baseline days (STATUS open issue 5). The guided miss R-014 returned brand=samsung for electronics & samsung (row Jaccard 0.95).

**Gate 3: passed.**
- The pipeline runs end to end on REES46, with memos for clean and injected data.
- Metrics files are written (`metrics.json`, `incidents.csv`).
- This section states what is real and what is injected.
- `evaluate` itself does not run on external data, by design (above).

## Phase 4: handover

### Gate 4: clean clone

Run on 2026-10-23, 23:46-23:54 local time, in a fresh directory outside the repo:

```bash
git clone --branch ticket-reporting <repo> gate4 && cd gate4      # HEAD c66ed09
uv sync --locked --extra dev --extra embeddings
.venv/Scripts/python -m pytest -p no:cacheprovider
.venv/Scripts/python -m kpi_rca generate --config config/default.yaml --out data
.venv/Scripts/python -m kpi_rca run --data data --out outputs
```

| Step | Result |
|---|---|
| clone | HEAD `c66ed09` |
| `uv sync --locked` | exit 0 (lock file is current) |
| pytest | `315 passed in 199.24s` |
| generate | 1,765,633 transactions, 798 tickets (seed 42) |
| run | W-001 kpi_drop 09-10 13:00-17:00 → INC-20260910-1300; W-002 volume_surge → no memo; W-003 kpi_drop 09-16 17:00-21:00 → INC-20260916-1700. Same as the working copy. 8 files in `outputs/reports` (md, json, kpi and shap png per memo) |

**Gate 4: passed.**

CI: `.github/workflows/ci.yml` exists (`uv sync --locked --extra dev`, then pytest). It has **not** run on GitHub, because the branch is not pushed yet. Pushing a workflow file needs `gh auth refresh -h github.com -s workflow`.

### Files changed or added on `ticket-reporting` (against `main`)

| File | Status | Why |
|---|---|---|
| `src/kpi_rca/reporting/llm.py` | changed | `OpenAICompatibleNarrator`: free, opt-in, JSON schema, retry with violations, template fallback |
| `src/kpi_rca/reporting/guardrails.py` | changed | Country names and known segment values come from `ReportingConfig`; wrong-segment check |
| `src/kpi_rca/reporting/impact.py` | changed | Lost revenue from the window's order value (scored against the new truth) |
| `src/kpi_rca/reporting/models.py` | changed | `narrator_used` may be `openai_compatible` |
| `src/kpi_rca/retrieval/query.py` | changed | Query attributes and customer wording come from `RetrievalConfig` |
| `src/kpi_rca/config.py` | changed | LLM fields (`base_url`, `openai_model`, `api_key_env`, `response_format`, `effort`, `temperature`); `query_attributes`, `customer_words`, `country_names`, `known_segment_values`; `dataset: generated \| external` |
| `src/kpi_rca/data/adapters.py` | **added** | REES46 → transaction schema, `no_tickets`, `candidate_segments`, `inject_incident` |
| `src/kpi_rca/data/io.py` | changed | Validation takes the attribute list (core columns + configured attributes) |
| `src/kpi_rca/data/generator.py` | changed | `lost_orders`, `lost_revenue` truth per incident |
| `src/kpi_rca/evaluation.py` | changed | Scores impact against the truth; A4 with the free narrator; passes the retrieval config |
| `src/kpi_rca/pipeline.py` | changed | Loads with the configured attributes; passes the retrieval config to the query |
| `src/kpi_rca/cli.py` | changed | `--narrator openai_compatible`; `generate` / `evaluate` refuse `dataset: external` |
| `config/default.yaml` | changed | New LLM settings (template stays the default) |
| `config/rees46.yaml` | **added** | REES46 attributes, `dataset: external` |
| `scripts/validate_rees46.py` | **added** | REES46 semi-synthetic scoring; `--export` for `run` |
| `tests/test_adapters.py` | **added** | 8 tests: mapping, validation, no tickets, injection on a copy, CLI refusal |
| `tests/test_guardrails_adversarial.py` | **added** | 18 adversarial guardrail tests |
| `tests/test_llm.py` | changed | Fake-client tests of the free narrator; A's `test_my_reporting_...` |
| `tests/test_retrieval.py` | changed | A's `test_my_retrieval_...`; config-driven query test |
| `tests/test_guardrails.py`, `test_config.py`, `test_data.py`, `test_reporting_impact.py` | changed | Configurable names, external config, loss truth, impact vs truth |
| `docs/VERIFICATION.md`, `docs/DEMO.md` | **added** | This report; the demo script |
| `docs/STATUS.md`, `README.md`, `docs/AI_USE_LOG.md`, `docs/PLAN.md`, `docs/RESEARCH.md` | changed | Final numbers, new commands, AI-use rows, REES46 reference |

`AGENT_PLAN.md` (repo root) is not committed; it is the user's brief.

### Remaining misses

| # | Miss | Number | Owner | State |
|---|---|---|---|---|
| 1 | Detection precision | 0.62 vs ≥ 0.80 | B | Conversion-only gate proposed (seed 101: 0.89); team decision |
| 2 | Detection F1 | 0.77 vs ≥ 0.85 | B | Follows from 1 |
| 3 | RCA top-3 | 0.89 vs ≥ 0.90 (one incident) | C | One-attribute extension fix described, not built |
| 4 | Guided vs brute-force exact gap | 8 points vs ≤ 5 | C | Same cause as 3 |
| 5 | LLM first-attempt pass | 0.00 vs ≥ 0.80 (gemma3:1b, local) | A | Only a 1B local model was available; free-tier API run needs a decision and key |
| 6 | LLM pass after retries | 0.20 vs ≥ 0.95 | A | As 5; the template fallback keeps memos compliant |
| 7 | Streaming F1 | not measured | B, D | Needs `stream_mode: full` |
| 8 | REES46 detection recall | 0.30 (0/10 for injected drops with z above −5) | B | Heavy-tailed real data (STATUS open issue 6); no target set for REES46 |
| 9 | REES46 lost revenue | −65% (USD 78,383 vs 221,918) on the injected apple incident | A | Window-wide order value; a segment-level order value would fix it; not built |
| 10 | REES46 RCA on level shifts | 2 of 6 detected incidents had no findings | C | Flat 7-day baseline (STATUS open issue 5) |
| 11 | REES46 detection stability | One injected incident removed 3 unrelated clean-run windows | B | Isolation Forest thresholds depend on the whole month |
| 12 | REES46 tickets | No ticket text exists, so retrieval and quotes are untested on real data | A | No real text source chosen; none fabricated |
| 13 | A3 distractor attributes, A4 raw-table arm, held-out run | not run | C, A, D | W5-W7 |
