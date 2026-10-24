# Project Status: KPI Root-Cause Analysis Engine

As of 2026-10-24, branch `ticket-reporting`. PR #4 merged it up to `c7702f0` into `main` (`c940318`); later commits are pushed but not merged. Measured numbers in sections 4-5 are from the 100-scenario run on a free Kaggle 2x T4 on 2026-10-24 (code `c7702f0`, LLM `gemma3:27b`); the 20-scenario numbers of 2026-10-23 (code `c66ed09`) are kept beside them. Scope and targets: [PLAN.md](PLAN.md). References: [RESEARCH.md](RESEARCH.md). Full verification run of AGENT_PLAN phases 0-4, with every command: [VERIFICATION.md](VERIFICATION.md). Demo script: [DEMO.md](DEMO.md).

**Legend:** `[x]` done, with evidence · `[~]` partly done · `[ ]` not started · **MISS** = built and measured, but the target is not met.
**Evidence** means something a team member can open or rerun: a file and line, a named test, or a command plus the output file it writes.

## 1. Overall progress

| Area | Done | Partial | Not started / missed | Progress |
|---|---|---|---|---|
| [2. Code modules](#2-code-modules) | 21 / 21 | 0 | 0 | 100% |
| [3. Tests](#3-tests) | 323 passing, 16 files | | | Done |
| [4. Success criteria](#4-success-criteria-plan-54) | 13 / 19 met | | 5 missed, 1 not measured | 68% |
| [5. Ablations](#5-ablations-plan-53) | 3 / 5 | 2 (A3, A4 context arm) | 0 | 80% |
| [6. Design review notes](#6-design-review-notes-plan-end) | 4 / 8 | 3 | 1 | 69% |
| [7. Project deliverables](#7-project-deliverables-plan-8-and-10) | 6 / 18 | 3 | 9 | 42% |
| [8. Timeline W1-W10](#8-timeline-w1-w10-plan-7) | 0 / 10 weeks closed | W4 (D) and W5-W6 (D) work exists from v1 | | W1 not yet closed |

Short version:
- The **software is built and tested**.
- The **numbers exist for the dev seed**, on 100 scenarios (159 incidents). Five targets are missed: detection precision and F1, surges labelled kpi_drop (0.02), BR blamed (0.01, one incident), and the guided-vs-brute-force gap (6 points). The last three were 0 or met on 20 scenarios, and the reverse holds for RCA top-3, which is now met (0.93). The two LLM guardrail rates are met by local `gemma3:27b` on 50 memos (0.90, 0.98).
- README, CI workflow, AI-use log, VERIFICATION and DEMO exist.
- REES46 runs end to end, with injected incidents.
- The **team work** (explainers, report, slides) has **not started**.

## 2. Code modules

All modules exist and have tests. "Built" means written by the AI-assisted v1; the owner still has to read, run and change it before W4 (ownership rule, PLAN §6).

| Done | Module | What it does | Evidence (tests) | Owner |
|---|---|---|---|---|
| [x] | `data/generator.py` | Synthetic transactions, tickets, ground truth | `tests/test_data.py` (17), e.g. `test_incident_forces_failures_only_inside_segment_and_time` | B |
| [x] | `data/io.py` | Parquet/JSON save, load, validation (core columns + the configured attributes) | `test_save_load_roundtrip`, `test_validate_transactions_rejects_bad_input` | B |
| [x] | `data/adapters.py` | REES46 event log → transaction schema; no tickets; incidents injected on a copy (2026-10-23) | `tests/test_adapters.py` (8), e.g. `test_injection_changes_a_copy_and_only_the_segment_in_range` | A (built), B to review |
| [x] | `detection/kpi.py` | Hourly KPIs | `test_aggregate_kpis_matches_transactions` | B |
| [x] | `detection/detector.py` | Robust STL, MAD z, IsolationForest, labels | `tests/test_detection.py` (16), e.g. `test_batch_never_calls_the_surge_a_drop` | B |
| [x] | `detection/streaming.py` | Past-only replay, delay | `test_streaming_never_looks_ahead`, `test_streaming_detects_the_incident_with_a_delay` | B |
| [x] | `rca/segment_model.py` | M0/M1 LightGBM, `init_score`, TreeSHAP | `test_shap_outputs_use_attribute_value_keys`, `test_high_baseline_country_is_not_blamed` | C |
| [x] | `rca/search.py` | SHAP-guided slice search, EP/purity, z-test | `test_shap_search_finds_the_incident_segment`, `test_slice_score_and_z_test_by_hand` | C |
| [x] | `rca/baseline.py` | Brute force up to 3 attributes | `test_brute_force_agrees_but_scores_more_slices` | C |
| [x] | `retrieval/query.py` | Segment + error codes as customer words; attributes and wording from `retrieval.query_attributes` / `customer_words` | `test_query_for_the_paypal_ios_incident`, `test_query_leaves_out_the_country`, `test_query_attributes_and_wording_come_from_the_config` | A |
| [x] | `retrieval/embeddings.py` | MiniLM or LSA, `auto` fallback | `test_auto_falls_back_to_lsa_with_a_warning` | A |
| [x] | `retrieval/index.py` | FAISS flat index, time filter, boosts, spike | `test_tickets_outside_the_padded_window_are_never_returned`, `test_spike_statistics_match_a_hand_computed_case` | A |
| [x] | `reporting/impact.py` | Lost revenue, lost orders, severity; scored against the generator's loss truth | `tests/test_reporting_impact.py` (17) | A |
| [x] | `reporting/context.py` | Compact JSON, the only LLM input | `tests/test_context.py` (15), e.g. `test_customer_text_appears_only_under_the_untrusted_excerpt_key` | A |
| [x] | `reporting/models.py` | Pydantic memo schema | Covered by `test_invalid_payload_reports_schema_violation` | A |
| [x] | `reporting/llm.py` | Template narrator (default); free OpenAI-compatible narrator (local Ollama by default, opt-in, 2026-10-23); Claude (paid, opt-in) | `tests/test_llm.py` (62), e.g. `test_falls_back_to_the_template_when_every_draft_fails` | A |
| [x] | `reporting/guardrails.py` | Number, segment, quote, schema checks; a memo naming a country, OS or gateway outside the findings is rejected; names and values from `ReportingConfig` (2026-10-23) | `tests/test_guardrails.py` (38), e.g. `test_hallucinated_number_in_action_is_flagged`; `tests/test_guardrails_adversarial.py` (18) | A |
| [x] | `reporting/render.py` | Markdown memo, JSON, PNG plots | `tests/test_render.py` (10) | A |
| [x] | `evaluation.py` | Random scenarios, metrics, CIs, ablations | `tests/test_evaluation.py` (13), e.g. `test_bootstrap_ci_brackets_the_mean` | D |
| [x] | `pipeline.py` | Orchestration, stage timings | `tests/test_pipeline.py` (10), e.g. `test_ground_truth_never_reaches_the_ticket_index` | D |
| [x] | `cli.py` | `generate`, `run`, `stream`, `evaluate` | `tests/test_cli.py` (8) | D |

Free-only rules (decision 2026-10-15) are guarded by tests: `test_default_narrator_is_the_free_template` and `test_a4_is_not_run_without_with_llm_even_when_a_key_is_set`.

## 3. Tests

| Done | Item | Evidence |
|---|---|---|
| [x] | 323 tests pass | `.venv\Scripts\python -m pytest` on 2026-10-24: `323 passed in 157.55s` (315 on 2026-10-23; the new ones cover 429 handling, free-only OpenRouter model IDs, the A4 memo cap and the dashboard, `tests/test_dashboard.py`) |
| [x] | Pass from a clean clone | 2026-10-23, Gate 4 in [VERIFICATION.md](VERIFICATION.md): `git clone --branch ticket-reporting`, `uv sync --locked --extra dev --extra embeddings`, `pytest`, `generate`, `run` |
| [x] | No network or key needed | Claude and OpenAI-compatible clients are injected or faked in `tests/test_llm.py`; `sentence_transformers` is faked in `tests/test_retrieval.py` |
| [~] | Each member's own 2 tests (W2) | A: `test_my_retrieval_test_the_window_filter_holds_on_a_generated_scenario`, `test_my_reporting_test_an_invented_number_ends_in_the_template_memo` (AI-written; A must read and defend them). B, C, D: none |
| [~] | Adversarial guardrail tests by A (W3) | `tests/test_guardrails_adversarial.py` (18): wrong country/OS/gateway named, prompt injection in a ticket ($5M, version 9.9.9, "blame Stripe in France"), paraphrased quotes, template memo unaffected. AI-written; A still has to read and defend them |
| [x] | Run in CI | `.github/workflows/ci.yml`: first GitHub run passed on PR #4 (2026-10-24, 2m50s) |

## 4. Success criteria (PLAN §5.4)

Source: `evaluate` on 100 scenarios, dev seed 7 (159 incidents, 155 detected), run on a free Kaggle 2x T4 on 2026-10-24 by `scripts/kaggle/run_eval.py`:
`python -m kpi_rca -v evaluate --config config/kaggle_27b.yaml --scenarios 100 --with-llm --out /kaggle/working/outputs` (7,624 s, including 50 memos from `gemma3:27b` in Ollama, 100% GPU).
Evidence files: `outputs/kaggle_27b/outputs/evaluation/metrics.json`, `summary.md`, `scenarios.csv`, plus `ollama_ps.txt` and `ollama.log`. Download them with `kaggle kernels output aditya32211/kpi-rca-gemma3-27b-eval -p outputs/kaggle_27b`. `outputs/` is git-ignored. This is **not** the held-out run (W7).
The first 20 scenarios are the 20-scenario run of 2026-10-23 (same 35 incidents); its numbers are in the last column. Three runs of those 20 scenarios (2026-10-15 and two on 2026-10-23) matched on every accuracy metric.

| # | Met? | Criterion | Target | 100 scenarios (95% CI) | 20 scenarios (2026-10-23) |
|---|---|---|---|---|---|
| 1 | [x] | Detection recall | ≥ 0.90 | 0.97 (0.95-0.99) | 1.00 |
| 2 | **MISS** | Detection precision | ≥ 0.80 | 0.52 (0.48-0.57); 144 of 299 kpi_drop windows match no incident | 0.62 (0.53-0.73) |
| 3 | **MISS** | Detection F1 | ≥ 0.85 | 0.68 (0.64-0.72) | 0.77 (0.70-0.85) |
| 4 | **MISS** | Surge never labelled kpi_drop | 0 | 0.02 | 0.00 |
| 5 | [x] | Streaming median delay | ≤ 3 buckets | 2.0; 0.94 of incidents detected | 2.0; 0.97 |
| 6 | [ ] | Streaming F1 | ≥ 0.85 | Not measured: needs `stream_mode: full` | |
| 7 | [x] | RCA top-3 | ≥ 0.90 | 0.93 (0.89-0.97), 144 of 155 | **MISS** 0.89 (0.77-0.97), 31 of 35 |
| 8 | [x] | RCA exact match | ≥ 0.75 | 0.92 (0.88-0.96), 143 of 155 | 0.86 (0.74-0.97) |
| 9 | [x] | RCA row-level Jaccard | ≥ 0.85 | 0.98 (0.96-0.99) | 0.94 (0.88-0.99) |
| 10 | **MISS** | BR never blamed | 0 | 0.01 (0.00-0.02): 1 of 155 (scenario 87, INC-01). The top finding was right; a later finding named country=BR | 0.00 |
| 11 | [x] | Guided slices vs brute force | ≤ 10% | 2.8% (2.8-2.9) | 2.8% |
| 12 | **MISS** | Guided exact within 5 points of brute force | ≤ 5 points | 0.92 vs 0.98 = 6 points | 0.86 vs 0.94 = 8 points |
| 13 | [x] | Retrieval P@5, MiniLM | ≥ 0.80 | 0.90 (0.87-0.93) | 0.91 (0.83-0.97) |
| 14 | [x] | Retrieval P@5, LSA | ≥ 0.60 | 0.78 (0.75-0.82) | 0.75 (0.69-0.81) |
| 15 | [x] | R@10 reported with its cap | report | `r_at_10` 0.55, cap 0.59 (MiniLM) | reported |
| 16 | [x] | 100% of final memos compliant | 1.00 | 1.00 (template); the one `gemma3:27b` memo that failed every attempt fell back to the template | 1.00 |
| 17 | [x] | LLM first-attempt pass | ≥ 0.80 | 0.90 (45 of 50 memos), local `gemma3:27b` | 0.80 (8 of 10), OpenRouter free `dots-3-note-preview`; local `gemma3:4b` 0.40, `gemma3:1b` 0.00 |
| 18 | [x] | LLM pass after retries | ≥ 0.95 | 0.98 (49 of 50), same run | 1.00 (10 of 10), dots-3; `gemma3:4b` 0.60, `gemma3:1b` 0.20 |
| 19 | [x] | End-to-end time, template | ≤ 60 s | 15.6 s (15.0-16.2) per incident, Kaggle CPU | 14.2 s (13.2-15.2) |

Also measured on 100 scenarios, no target:
- 0.14 false kpi_drop windows per day on the incident-free run.
- Brute-force top-3 is 0.99, exact 0.98.
- Recall by visibility bin: 0.96, 0.96, 1.00.
- Guided RCA is 2.85x **slower** in wall time than brute force (15.9 s vs 5.2 s median).
- Impact against the generator's loss truth: mean relative error +0.11 (0.04 to 0.17), median absolute 0.14 (155 incidents). On 20 scenarios it was +0.01 (−0.08 to 0.10), so lost revenue leans high.
- `gemma3:27b` takes a median 51 s per memo (first 57 s, which includes loading 18 GB onto the two T4s).

Five times the scenarios narrowed the CIs, and three verdicts changed: criteria 4 and 10 now miss, and 7 is met. The 10-memo runs of the other narrators are in [VERIFICATION.md](VERIFICATION.md), "LLM narrators compared".

## 5. Ablations (PLAN §5.3)

100 scenarios (section 4) unless marked; 20-scenario numbers in brackets.

| Done | ID | What | Result | Evidence |
|---|---|---|---|---|
| [x] | A1 | MiniLM vs LSA | P@5 0.90 vs 0.78; P@10 0.87 vs 0.80 (20: 0.91 vs 0.75) | `summary.md`, "Retrieval" table |
| [x] | A2 | M1 without the M0 offset | Without the offset: found 0.79, exact 0.70 (20: 0.69, 0.57). BR blamed 0.01 with and without the offset; median BR rank in feature importance 16 with, 2 without | `metrics.json`, `rca.no_offset`, `rca.a2_offset` |
| [~] | A3 | Guided vs brute force + distractor attributes | Guided vs brute force done (2.8% of slices). **Distractors not run** | `metrics.json` says "not run (W5)" |
| [~] | A4 | LLM on raw table vs pipeline context | Context arm, first-attempt / after retries / fallback: local `gemma3:27b` 0.90 / 0.98 / 0.02 (50 memos). 10 memos on 20 scenarios: OpenRouter free `dots-3-note-preview` 0.80 / 1.00 / 0.00; local `gemma3:4b` 0.40 / 0.60 / 0.40; local `gemma3:1b` 0.00 / 0.20 / 0.80. **Raw-table arm not built** | `metrics.json`, `a4_llm`; memo count is `evaluation.a4_max_memos` |
| [x] | A5 | Metadata boosts off | P@5 MiniLM 0.85, LSA 0.75 (20: 0.83, 0.70) | `metrics.json`, retrieval no-boost |

## 6. Design review notes (PLAN, end)

| Done | Note | State | Evidence |
|---|---|---|---|
| [x] | 1. Equivalent (nested) slices | Redundant segments kept out of scenario sampling; Jaccard is row-level | `test_is_redundant`, `test_catalogue_respects_the_share_bounds_and_drops_redundant_segments` |
| [~] | 2. Spurious findings | Bonferroni α / slices scored, same rule in both searches. **Extra findings in single-incident scenarios not counted** | `rca/search.py:144`, `rca/baseline.py:113` |
| [~] | 3. Detection | Surges kept 12 h from incidents; incident-free run exists. **MSTL not re-measured** (B, W3) | `evaluation.py:60` (`MIN_GAP_H`), `incident_free_runs` in `config/default.yaml:190` |
| [x] | 4. Lost-revenue truth | `ground_truth.json` has `lost_orders` and `lost_revenue` per incident; `evaluate` scores `impact.py` (median abs. error 0.11 on 20 scenarios, 0.14 on 100) | `data/generator.py`, `tests/test_data.py`, `tests/test_reporting_impact.py` |
| [x] | 5. LLM settings | Refusal and cut-off handled; `LLMConfig.effort` added (sent as `reasoning_effort` only when set) | `test_a_refusal_or_cut_off_falls_back_without_retrying`, `tests/test_llm.py` |
| [x] | 6. Retrieval metrics | Evaluation retrieves `max(10, relevant)`; R@10 cap and R-precision reported; A5 runs | `evaluation.py:290-297`, `evaluation.py:438` |
| [~] | 7. A4 scoring | Context arm scored by the guardrails (pass rates above); raw-table arm not built | `evaluation.py`, `_run_a4` |
| [ ] | 8. Raw-rate brute force (optional) | Not built | |

## 7. Project deliverables (PLAN §8 and §10)

| Done | Item | Owner | State / evidence |
|---|---|---|---|
| [x] | GitHub repo | A | `origin` = `github.com/nishida09datkar/DATA-SCIENCE-CP`, `main` pushed |
| [x] | Plan | A | `docs/PLAN.md` (team still has to review and adopt it in W1) |
| [x] | Research and references | All | `docs/RESEARCH.md`; some entries marked **[unverified]** |
| [x] | Protected `main`, PR + review rule | A | Branch protection since 2026-10-24 (GitHub API): PR with 1 approval, required check `test`, no force-push or delete; admins can bypass |
| [~] | Every reference verified at its source | All | Several **[unverified]** entries remain in RESEARCH.md |
| [x] | README | D | `README.md`: setup, run, outputs, config, cost, layout, contributing, Windows notes; install steps checked on a clean copy (section 3). A newcomer still has to try it in W4 |
| [x] | CI (GitHub Actions, windows-latest, Python 3.12) | D | `.github/workflows/ci.yml`; first GitHub run passed on PR #4 (2026-10-24) |
| [ ] | Project board, weekly sync | A | None found |
| [x] | AI-use log | All | `docs/AI_USE_LOG.md` started, with the four AI-assisted sessions so far; each member adds their own rows |
| [ ] | One explainer per member (W2) | All | None |
| [ ] | Tagged release (W7 freeze) | A | `git tag` is empty |
| [ ] | Held-out evaluation run (W7) | D | Not run; only dev seed 7 (20 and 100 scenarios) and a seed-101 precision check. `scripts/kaggle/` can run it on a free GPU |
| [ ] | `outputs/reports/INC-001.*`, `INC-002.*` from the tagged run | D | `run` works and the INC-001 memo is in [DEMO.md](DEMO.md), but no tagged run exists |
| [x] | Verification report and demo script | A | [VERIFICATION.md](VERIFICATION.md) (AGENT_PLAN phases 0-4, 19 criteria, REES46 real vs injected); [DEMO.md](DEMO.md). D still has to rehearse the demo |
| [~] | Semi-synthetic validation on REES46 data (W8, optional) | B, D | First run 2026-10-23 on Oct 2019 (910,796 attempts, 31 days): `.venv\Scripts\python scripts\validate_rees46.py --month 2019-Oct --incidents 20` writes `outputs/rees46/2019-Oct/metrics.json` and `incidents.csv`. Clean month: 0.23 kpi_drop windows/day (STL), 0.16 (MSTL forced; unlabelled, may be real). Injected: detection 6/20 (6/10 with min conversion z ≤ −5, 0/10 above −3); RCA on the 4 windows with positive excess: guided exact 3/4 (miss = `brand=samsung` for `electronics & samsung`, row Jaccard 0.95), brute force 4/4; 2 detected windows had no findings (open issue 5). Reproduced exactly after the move onto `data/adapters.py`. **`run` on REES46** (`config/rees46.yaml`): the clean month gives 7 unexplained kpi_drop memos. With one injected incident (brand=apple, 10-25 18:00-23:00, 240 failures) the window, the segment and a memo come out right, but lost revenue is −65% (VERIFICATION.md, Phase 3). REES46 has no ticket text; none is fabricated. Owners still have to rerun and explain it |
| [ ] | Report (7 chapters + appendices) | All | Not started |
| [ ] | Slides (12) | All | Not started; outline in PLAN §10.3 |
| [ ] | Demo rehearsal and backup video | D | Not started; script in PLAN §10.4 |
| [ ] | Mock viva | All | Question bank exists (PLAN §10.5); no rehearsal |

## 8. Timeline W1-W10 (PLAN §7)

Week dates are not fixed yet (map back from the deadline). No week is closed.

| Week | Exit check | A | B | C | D |
|---|---|---|---|---|---|
| W1 Set up | Everyone has run all commands | [~] repo, AI log, all commands run (VERIFICATION.md); board no | [ ] | [ ] | [~] README, CI file written (v1) |
| W2 Understand | Explainer + 2 tests each | [ ] | [ ] | [ ] | [ ] |
| W3 Validate | Targets confirmed and frozen | [~] adversarial guardrail tests written (AI, A to review); [ ] retrieval label sanity | [ ] STL plots, MSTL rerun | [ ] M0 calibration, brute-force agreement | [ ] |
| W4 Checkpoint 1 | Clean-clone reproduction by all | [ ] | [ ] | [ ] | [~] 20- and 100-scenario dev runs exist (v1) |
| W5 Extend | | [~] A1, A5 run (v1) | [ ] incident-free, red-team, distractor scenarios | [~] A2 runs (v1) | [ ] |
| W6 Experiment | | [~] template guardrail run = 1.00 (v1) | [~] recall by visibility in `summary.md` (v1) | [ ] A3 scaling curve | [~] CIs in `evaluate` (v1) |
| W7 Checkpoint 2 | Tag, frozen config, held-out numbers | [ ] | [ ] | [ ] | [ ] |
| W8 Buffer | | [ ] figures | [~] REES46 semi-synthetic (with D): first run exists | [ ] literature baseline | [~] REES46 semi-synthetic (with B): first run exists |
| W9 Write | | [ ] | [ ] | [ ] | [ ] |
| W10 Submit | | [ ] | [ ] | [ ] | [ ] |

`[~]` in W4-W6 means the v1 code already produces it; it counts only once the owner has rerun and can explain it.

## 9. Open issues and decisions needed

1. **Detection precision (B).** 21 of 56 kpi_drop windows match no incident on 20 scenarios, 144 of 299 on 100 (precision 0.52): traffic and revenue dip together while conversion is normal. On 100 scenarios 2% of surges are also labelled kpi_drop (criterion 4). A conversion-only gate raised seed-101 precision to 0.89 with recall unchanged, but redefines kpi_drop. Team decision needed (GitHub issue #1). Evidence: PLAN §5.5, "Detection precision miss".
2. **RCA top-3 and brute-force gap (C).** Correlated or nested attributes (mobile within Android) drop out of the candidate list; this causes the 8-point gap to brute force on 20 scenarios (6 points on 100) and the top-3 miss on 20 (top-3 is met on 100: 0.93). Fix (score one-attribute extensions of the chosen slice) is described, not built. Evidence: PLAN §5.5, "RCA top-3 miss". On 100 scenarios one incident (scenario 87, INC-01) also had country=BR in a later finding (criterion 10); `scenarios.csv` has the row.
3. **Not built:** A3 distractor attributes; A4 raw-table arm; held-out run.
4. **Free LLM (A, GitHub issue #2).** The OpenAI-compatible narrator is built. Measured on 2026-10-24 (first-attempt / after retries):
   - Local `gemma3:27b` on a free Kaggle 2x T4 (`scripts/kaggle/`, `config/kaggle_27b.yaml`): 0.90 / 0.98 on 50 memos; 1 memo fell back to the template. Meets criteria 17-18 with no key and no data leaving the machine; it needs about 18 GB of GPU memory, so it cannot run on the team's laptops.
   - OpenRouter free `dots-studio/dots-3-note-preview:free`: 0.80 / 1.00. Meets criteria 17-18. It is a reasoning model: it needs `max_tokens: 16000` and `effort: low`, or it stops at `finish_reason=length`. The first choice, `nvidia/nemotron-3-super-120b-a12b:free`, answered 404 from its provider. Free providers may log prompts, so it gets synthetic data only.
   - Local `gemma3:4b` (6 GB GPU, `config/ollama_4b.yaml`): 0.40 / 0.60. Local `gemma3:1b`: 0.00 / 0.20 (quotes wrapped in extra quotation marks, too many quotes or actions, segment values left out).
   - Team decision: which narrator the demo and report use (template default; 27b on Kaggle, dots-3 on OpenRouter, or 4b locally).
   - `config/groq.yaml` (Groq free tier) is still untested: no key.
   - Also decide whether the quote check should ignore surrounding quotation marks.
5a. **Impact on segments with unusual baskets (A).** Lost revenue = lost orders × the window's average order value. That gives a mean error of +0.01 on synthetic data, but −65% on the REES46 apple incident, whose baskets are far above average. A segment-level order value would fix it; not built (GitHub issue #3).
5. **RCA baseline ignores level shifts (C).** On REES46, R-002 and R-015 were detected but got no findings from either method: the 7-day baseline fell in days with heavier cart logging (failure rate 0.34 vs 0.28 in the window), so the window's total excess over M0 was negative (−103, −192) and RCA stops at E ≤ 0. Detection compares with an hourly STL expectation; RCA compares with a flat 7-day average. Synthetic data has no such shift. Evidence: `rca_excess` column in `outputs/rees46/2019-Oct/incidents.csv`; `brute_force_window` logs "has no excess failures".
6. **Detection on heavy-tailed real data (B).** The clean October already has 7 buckets with conversion z < −5 and 14 with |attempts z| > 5 (cart-logging jumps on Oct 4-6 and 14-27), so the Isolation Forest does not isolate injected drops of z −5 to −6 (R-003, R-005, R-013, R-019 missed). Evidence: `min_conversion_z` column in `incidents.csv`.
7. **Review needed (B, C, D): the schema is now config-driven.**
   - `RCAConfig.attributes` accepts any column names.
   - `AppConfig.dataset: external` lets the whole pipeline run on them; `generated` (the default) still limits it to `SEGMENT_ATTRIBUTES`.
   - `validate_transactions` checks the core columns plus the configured attributes.
   - Interface change per PLAN §8. Tests: `test_rca_attributes_are_free_alone_but_generated_in_the_app`, `tests/test_adapters.py`.
8. **Detection on REES46 moves with distant data (B).** Injecting one incident on 10-25 removed three unrelated clean-run windows. Isolation Forest thresholds depend on the whole month.

## 10. Next steps by member

| Member | Now (W1-W3) |
|---|---|
| **A, lead: retrieval and reporting** | Read and defend the AI-written narrator, adapter and tests (above all the two `test_my_` tests); start the board; table decisions 1, 4 and 5a |
| **B, data and detection** | Run all commands; explainer + 2 tests; recommend on the precision gate; STL plots; STL vs MSTL rerun |
| **C, RCA** | Run all commands; explainer + 2 tests; build and measure the top-3 fix; M0 calibration; brute-force agreement |
| **D, integration and evaluation** | Run all commands; explainer + 2 tests; reproduce `metrics.json` from a clean clone; prepare A3 distractor runs; demo script |

## 11. Decisions log

| Date | Decision |
|---|---|
| 2026-10-15 | Free product: template narrator is the default; no paid API in runs, tests, demos or evaluation |
| 2026-10-19 | A4 with the paid API is out of scope; free-model run proposed, pending team decision |
| 2026-10-23 | Roles re-split: A = ticket retrieval and reporting; D = integration and evaluation (pipeline, CLI, evaluation, docs, CI, demo); B and C unchanged. PLAN §6 updated |
| 2026-10-23 | Lead adopts the REES46 Kaggle dataset for W8 semi-synthetic validation (detection and RCA only; one month); PLAN §3.3 |
