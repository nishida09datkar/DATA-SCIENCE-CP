# Project Status: KPI Root-Cause Analysis Engine

As of 2026-10-23, commit `b7a355c` (pushed to `origin/main`). Scope and targets: [PLAN.md](PLAN.md). References: [RESEARCH.md](RESEARCH.md).

**Legend:** `[x]` done, with evidence · `[~]` partly done · `[ ]` not started · **MISS** = built and measured, but the target is not met.
**Evidence** means something a team member can open or rerun: a file and line, a named test, or a command plus the output file it writes.

## 1. Overall progress

| Area | Done | Partial | Not started / missed | Progress |
|---|---|---|---|---|
| [2. Code modules](#2-code-modules) | 20 / 20 | 0 | 0 | 100% |
| [3. Tests](#3-tests) | 261 passing, 13 files | | | Done |
| [4. Success criteria](#4-success-criteria-plan-54) | 12 / 19 met | | 4 missed, 3 not measured | 63% |
| [5. Ablations](#5-ablations-plan-53) | 3 / 5 | 1 (A3) | 1 (A4, out of scope) | 70% |
| [6. Design review notes](#6-design-review-notes-plan-end) | 2 / 8 | 4 | 2 | 50% |
| [7. Project deliverables](#7-project-deliverables-plan-8-and-10) | 5 / 17 | 3 | 9 | 38% |
| [8. Timeline W1-W10](#8-timeline-w1-w10-plan-7) | 0 / 10 weeks closed | W4 (D) and W5-W6 (D) work exists from v1 | | W1 not yet closed |

Short version: the **software is built and tested**; the **numbers exist for the dev seed** (four targets missed); README, CI workflow and AI-use log now exist; the **team work** (explainers, validation, report, slides) has **not started**.

## 2. Code modules

All modules exist and have tests. "Built" means written by the AI-assisted v1; the owner still has to read, run and change it before W4 (ownership rule, PLAN §6).

| Done | Module | What it does | Evidence (tests) | Owner |
|---|---|---|---|---|
| [x] | `data/generator.py` | Synthetic transactions, tickets, ground truth | `tests/test_data.py` (17), e.g. `test_incident_forces_failures_only_inside_segment_and_time` | B |
| [x] | `data/io.py` | Parquet/JSON save, load, validation | `test_save_load_roundtrip`, `test_validate_transactions_rejects_bad_input` | B |
| [x] | `detection/kpi.py` | Hourly KPIs | `test_aggregate_kpis_matches_transactions` | B |
| [x] | `detection/detector.py` | Robust STL, MAD z, IsolationForest, labels | `tests/test_detection.py` (16), e.g. `test_batch_never_calls_the_surge_a_drop` | B |
| [x] | `detection/streaming.py` | Past-only replay, delay | `test_streaming_never_looks_ahead`, `test_streaming_detects_the_incident_with_a_delay` | B |
| [x] | `rca/segment_model.py` | M0/M1 LightGBM, `init_score`, TreeSHAP | `test_shap_outputs_use_attribute_value_keys`, `test_high_baseline_country_is_not_blamed` | C |
| [x] | `rca/search.py` | SHAP-guided slice search, EP/purity, z-test | `test_shap_search_finds_the_incident_segment`, `test_slice_score_and_z_test_by_hand` | C |
| [x] | `rca/baseline.py` | Brute force up to 3 attributes | `test_brute_force_agrees_but_scores_more_slices` | C |
| [x] | `retrieval/query.py` | Segment + error codes as customer words | `test_query_for_the_paypal_ios_incident`, `test_query_leaves_out_the_country` | D |
| [x] | `retrieval/embeddings.py` | MiniLM or LSA, `auto` fallback | `test_auto_falls_back_to_lsa_with_a_warning` | D |
| [x] | `retrieval/index.py` | FAISS flat index, time filter, boosts, spike | `test_tickets_outside_the_padded_window_are_never_returned`, `test_spike_statistics_match_a_hand_computed_case` | D |
| [x] | `reporting/impact.py` | Lost revenue, lost orders, severity | `tests/test_reporting_impact.py` (10) | A |
| [x] | `reporting/context.py` | Compact JSON, the only LLM input | `tests/test_context.py` (15), e.g. `test_customer_text_appears_only_under_the_untrusted_excerpt_key` | A |
| [x] | `reporting/models.py` | Pydantic memo schema | Covered by `test_invalid_payload_reports_schema_violation` | A |
| [x] | `reporting/llm.py` | Template narrator (default), Claude (opt-in) | `tests/test_llm.py` (25), e.g. `test_falls_back_to_the_template_when_every_draft_fails` | A |
| [x] | `reporting/guardrails.py` | Number, segment, quote, schema checks | `tests/test_guardrails.py` (37), e.g. `test_hallucinated_number_in_action_is_flagged` | A |
| [x] | `reporting/render.py` | Markdown memo, JSON, PNG plots | `tests/test_render.py` (10) | A |
| [x] | `evaluation.py` | Random scenarios, metrics, CIs, ablations | `tests/test_evaluation.py` (13), e.g. `test_bootstrap_ci_brackets_the_mean` | D |
| [x] | `pipeline.py` | Orchestration, stage timings | `tests/test_pipeline.py` (10), e.g. `test_ground_truth_never_reaches_the_ticket_index` | A |
| [x] | `cli.py` | `generate`, `run`, `stream`, `evaluate` | `tests/test_cli.py` (8) | A |

Free-only rules (decision 2026-10-15) are guarded by tests: `test_default_narrator_is_the_free_template` and `test_a4_is_not_run_without_with_llm_even_when_a_key_is_set`.

## 3. Tests

| Done | Item | Evidence |
|---|---|---|
| [x] | 261 tests pass | `.venv\Scripts\python -m pytest -q` on 2026-10-23: 261 passed |
| [x] | Pass from a clean copy without the embeddings extra | 2026-10-23: `uv sync --locked --extra dev` then `uv run --no-sync pytest` gave `261 passed in 135.83s` (the README and CI steps) |
| [x] | No network or key needed | Claude client is injected or faked in `tests/test_llm.py`; `sentence_transformers` is faked in `tests/test_retrieval.py` |
| [ ] | Each member's own 2 tests (W2) | None yet; every test so far is from the v1 |
| [ ] | Adversarial guardrail tests by A (W3) | Not written; existing guardrail tests are the v1's |
| [~] | Run in CI | `.github/workflows/ci.yml` written; not yet run on GitHub (runs on the first push) |

## 4. Success criteria (PLAN §5.4)

Source: `evaluate` with the default config, 20 scenarios, dev seed 7. Evidence file: `outputs/evaluation/metrics.json` and `summary.md`, regenerated on 2026-10-23 by
`.venv\Scripts\python -m kpi_rca evaluate --scenarios 20 --out outputs` (the command adds `evaluation/` itself).
`outputs/` is git-ignored, so the files are local; anyone can regenerate them with that command (848 s on 2026-10-23). This is **not** the held-out run (W7).
Reproducibility check: the 2026-10-23 rerun matched the 2026-10-15 run on every accuracy metric. Only timings changed (the machine was also running the test suite), so the timings below are from 2026-10-23.

| # | Met? | Criterion | Target | Result (95% CI) |
|---|---|---|---|---|
| 1 | [x] | Detection recall | ≥ 0.90 | 1.00 |
| 2 | **MISS** | Detection precision | ≥ 0.80 | 0.62 (0.53-0.73) |
| 3 | **MISS** | Detection F1 | ≥ 0.85 | 0.77 (0.70-0.85) |
| 4 | [x] | Surge never labelled kpi_drop | 0 | 0.00 |
| 5 | [x] | Streaming median delay | ≤ 3 buckets | 2.0; 0.97 of incidents detected |
| 6 | [ ] | Streaming F1 | ≥ 0.85 | Not measured: needs `stream_mode: full` |
| 7 | **MISS** | RCA top-3 | ≥ 0.90 | 0.89 (0.77-0.97), 31 of 35 |
| 8 | [x] | RCA exact match | ≥ 0.75 | 0.86 (0.74-0.97) |
| 9 | [x] | RCA row-level Jaccard | ≥ 0.85 | 0.94 (0.88-0.99) |
| 10 | [x] | BR never blamed | 0 | 0.00 |
| 11 | [x] | Guided slices vs brute force | ≤ 10% | 2.8% |
| 12 | **MISS** | Guided exact within 5 points of brute force | ≤ 5 points | 0.86 vs 0.94 = 8 points. (Earlier STATUS and PLAN §5.5 said "yes"; this criterion was not checked.) |
| 13 | [x] | Retrieval P@5, MiniLM | ≥ 0.80 | 0.91 (0.83-0.97) |
| 14 | [x] | Retrieval P@5, LSA | ≥ 0.60 | 0.75 (0.69-0.81) |
| 15 | [x] | R@10 reported with its cap | report | `r_at_10` and `r_at_10_cap` in `metrics.json` (`evaluation.py:296-297`) |
| 16 | [x] | 100% of final memos compliant | 1.00 | 1.00 (template); `test_template_memos_always_pass_the_guardrails` |
| 17 | [ ] | LLM first-attempt pass | ≥ 0.80 | Not run: paid API out of scope |
| 18 | [ ] | LLM pass after retries | ≥ 0.95 | Not run: paid API out of scope |
| 19 | [x] | End-to-end time, template | ≤ 60 s | 11.2 s (10.3-12.0) per incident |

Also measured, no target: false kpi_drop windows 0.14 per day (incident-free run); brute-force top-3 0.97; recall 1.00 in every visibility bin; guided RCA is 2.5x **slower** in wall time than brute force (10.1 s vs 4.1 s median).

## 5. Ablations (PLAN §5.3)

| Done | ID | What | Result | Evidence |
|---|---|---|---|---|
| [x] | A1 | MiniLM vs LSA | P@5 0.91 vs 0.75; P@10 0.89 vs 0.80 | `summary.md`, "Retrieval" table |
| [x] | A2 | M1 without the M0 offset | BR blamed 0.00 with offset; without it found 0.69, exact 0.57 | `metrics.json`, RCA no-offset |
| [~] | A3 | Guided vs brute force + distractor attributes | Guided vs brute force done (2.8% of slices). **Distractors not run** | `metrics.json` says "not run (W5)" |
| [ ] | A4 | LLM on raw table vs pipeline context | **Out of scope** (paid API). Raw-table arm not built | `metrics.json` says "not run (paid API; ...)"; free-model proposal in PLAN §5.3 |
| [x] | A5 | Metadata boosts off | P@5 MiniLM 0.83, LSA 0.70 | `metrics.json`, retrieval no-boost |

## 6. Design review notes (PLAN, end)

| Done | Note | State | Evidence |
|---|---|---|---|
| [x] | 1. Equivalent (nested) slices | Redundant segments kept out of scenario sampling; Jaccard is row-level | `test_is_redundant`, `test_catalogue_respects_the_share_bounds_and_drops_redundant_segments` |
| [~] | 2. Spurious findings | Bonferroni α / slices scored, same rule in both searches. **Extra findings in single-incident scenarios not counted** | `rca/search.py:144`, `rca/baseline.py:113` |
| [~] | 3. Detection | Surges kept 12 h from incidents; incident-free run exists. **MSTL not re-measured** (B, W3) | `evaluation.py:60` (`MIN_GAP_H`), `incident_free_runs` in `config/default.yaml:190` |
| [ ] | 4. Lost-revenue truth | `ground_truth.json` still has no true lost amount, so `impact.py` cannot be scored | `data/generator.py` |
| [~] | 5. LLM settings | Refusal and cut-off handled. **No `effort` field** in `LLMConfig` (matters only if Claude is ever run) | `test_a_refusal_or_cut_off_falls_back_without_retrying` |
| [x] | 6. Retrieval metrics | Evaluation retrieves `max(10, relevant)`; R@10 cap and R-precision reported; A5 runs | `evaluation.py:290-297`, `evaluation.py:438` |
| [~] | 7. A4 scoring | Not applicable while A4 is not built | |
| [ ] | 8. Raw-rate brute force (optional) | Not built | |

## 7. Project deliverables (PLAN §8 and §10)

| Done | Item | Owner | State / evidence |
|---|---|---|---|
| [x] | GitHub repo | A | `origin` = `github.com/nishida09datkar/DATA-SCIENCE-CP`, `main` pushed |
| [x] | Plan | A | `docs/PLAN.md` (team still has to review and adopt it in W1) |
| [x] | Research and references | All | `docs/RESEARCH.md`; some entries marked **[unverified]** |
| [~] | Protected `main`, PR + review rule | A | Not verified on GitHub; both commits went straight to `main` |
| [~] | Every reference verified at its source | All | Several **[unverified]** entries remain in RESEARCH.md |
| [x] | README | A | `README.md`: setup, run, outputs, config, cost, layout, contributing, Windows notes; install steps checked on a clean copy (section 3). A newcomer still has to try it in W4 |
| [~] | CI (GitHub Actions, windows-latest, Python 3.12) | A | `.github/workflows/ci.yml` written and its steps pass locally; first GitHub run pending the push |
| [ ] | Project board, weekly sync | A | None found |
| [x] | AI-use log | All | `docs/AI_USE_LOG.md` started, with the four AI-assisted sessions so far; each member adds their own rows |
| [ ] | One explainer per member (W2) | All | None |
| [ ] | Tagged release (W7 freeze) | A | `git tag` is empty |
| [ ] | Held-out evaluation run (W7) | D | Not run; only dev seed 7 (and a seed-101 precision check) |
| [ ] | `outputs/reports/INC-001.*`, `INC-002.*` from the tagged run | A | `run` works (`test_run_prints_the_window_table_and_memos`), but no tagged run exists |
| [ ] | Report (7 chapters + appendices) | All | Not started |
| [ ] | Slides (12) | All | Not started; outline in PLAN §10.3 |
| [ ] | Demo rehearsal and backup video | A | Not started; script in PLAN §10.4 |
| [ ] | Mock viva | All | Question bank exists (PLAN §10.5); no rehearsal |

## 8. Timeline W1-W10 (PLAN §7)

Week dates are not fixed yet (map back from the deadline). No week is closed.

| Week | Exit check | A | B | C | D |
|---|---|---|---|---|---|
| W1 Set up | Everyone has run all commands | [~] repo, README, CI file, AI log yes; board no | [ ] | [ ] | [ ] |
| W2 Understand | Explainer + 2 tests each | [ ] | [ ] | [ ] | [ ] |
| W3 Validate | Targets confirmed and frozen | [ ] adversarial tests | [ ] STL plots, MSTL rerun | [ ] M0 calibration, brute-force agreement | [ ] retrieval label sanity |
| W4 Checkpoint 1 | Clean-clone reproduction by all | [ ] | [ ] | [ ] | [~] 20-scenario dev run exists (v1) |
| W5 Extend | | [ ] | [ ] incident-free, red-team, distractor scenarios | [~] A2 runs (v1) | [~] A1, A5 run (v1) |
| W6 Experiment | | [~] template guardrail run = 1.00 (v1) | [~] recall by visibility in `summary.md` (v1) | [ ] A3 scaling curve | [~] CIs in `evaluate` (v1) |
| W7 Checkpoint 2 | Tag, frozen config, held-out numbers | [ ] | [ ] | [ ] | [ ] |
| W8 Buffer | | [ ] | [ ] | [ ] | [ ] |
| W9 Write | | [ ] | [ ] | [ ] | [ ] |
| W10 Submit | | [ ] | [ ] | [ ] | [ ] |

`[~]` in W4-W6 means the v1 code already produces it; it counts only once the owner has rerun and can explain it.

## 9. Open issues and decisions needed

1. **Detection precision (B).** 21 of 56 kpi_drop windows match no incident: traffic and revenue dip together while conversion is normal. A conversion-only gate raised seed-101 precision to 0.89 with recall unchanged, but redefines kpi_drop. Team decision needed. Evidence: PLAN §5.5, "Detection precision miss".
2. **RCA top-3 and brute-force gap (C).** Correlated or nested attributes (mobile within Android) drop out of the candidate list; this causes both the top-3 miss and the 8-point gap to brute force. Fix (score one-attribute extensions of the chosen slice) is described, not built. Evidence: PLAN §5.5, "RCA top-3 miss".
3. **Not built:** A3 distractor attributes; A4 raw-table arm; held-out run; lost-revenue ground truth.
4. **Free LLM proposal (A, D).** Run A4 and the LLM guardrail runs on official free tiers through one OpenAI-compatible narrator. PLAN §5.3. Team decision needed.

## 10. Next steps by member

| Member | Now (W1-W3) |
|---|---|
| **A, lead** | Push and confirm the first green CI run; set branch protection; start the board; table decisions 1 and 4; adversarial guardrail tests |
| **B, data and detection** | Run all commands; explainer + 2 tests; recommend on the precision gate; STL plots; STL vs MSTL rerun |
| **C, RCA** | Run all commands; explainer + 2 tests; build and measure the top-3 fix; M0 calibration; brute-force agreement |
| **D, retrieval and evaluation** | Run all commands; explainer + 2 tests; reproduce `metrics.json` from a clean clone; prepare A3 distractor runs |

## 11. Decisions log

| Date | Decision |
|---|---|
| 2026-10-15 | Free product: template narrator is the default; no paid API in runs, tests, demos or evaluation |
| 2026-10-19 | A4 with the paid API is out of scope; free-model run proposed, pending team decision |
