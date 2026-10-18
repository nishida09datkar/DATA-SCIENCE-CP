# Project Status: KPI Root-Cause Analysis Engine

As of 2026-10-19. Source of truth for scope and targets: [PLAN.md](PLAN.md). Numbers below come from `evaluate` with the default config, 20 scenarios, development seed 7 (2026-10-15); they are not the held-out results.

## In one paragraph

An online shop's checkout sales drop. The engine finds when it happened (detection), which customers are affected, such as Android users on app 5.3.0 (root-cause analysis), which support tickets describe it (retrieval), and writes a 30-second memo with the lost revenue and fixes (reporting). Every number is computed deterministically; the memo text comes from a free template, and any LLM is optional and checked by guardrails.

## Build state

| Item | State |
|---|---|
| Code | AI-assisted v1 complete: data, detection, RCA, retrieval, reporting, CLI, evaluation |
| Tests | 261 passing |
| Commits | 1 (`3bb33f4`); free-only changes and plan updates not yet committed |
| Cost | Free: template narrator is the default; paid Claude path is opt-in only and never run |

## Results vs targets (dev seed 7)

| Area | Result | Target | Met? |
|---|---|---|---|
| Detection (batch) | Recall 1.00; precision 0.62; F1 0.77 | ≥ 0.90; ≥ 0.80; ≥ 0.85 | Recall yes; **precision, F1 no** |
| Detection (streaming) | Median delay 2.0 buckets | ≤ 3 | Yes (F1 not measured) |
| RCA guided | Exact 0.86; top-3 0.89; Jaccard 0.94 | ≥ 0.75; ≥ 0.90; ≥ 0.85 | **Top-3 no** (31/35) |
| RCA cost | 2.8% of brute-force evaluations | ≤ 10% | Yes (but 2.6x slower in wall time) |
| Retrieval P@5 | MiniLM 0.91; LSA 0.75 | ≥ 0.80; ≥ 0.60 | Yes |
| Guardrails (template) | Pass rate 1.00 | 1.00 | Yes |
| End-to-end (template) | 9.5 s per incident | ≤ 60 s | Yes |
| A4, LLM guardrails | Not run | — | Paid API out of scope |

## Open issues

1. **Detection precision (B).** 21 of 56 kpi_drop windows match no incident: traffic and revenue dip together while conversion is normal. A conversion-only gate raised seed-101 precision to 0.89 with recall unchanged, but redefines kpi_drop. Team decision needed.
2. **RCA top-3 (C).** Correlated or nested attributes (mobile within Android) drop out of the candidate list. A fix (score one-attribute extensions of the chosen slice) is described, not built.
3. **Not built:** A3 distractor attributes; A4 raw-table arm; held-out evaluation run (W7).
4. **Free LLM proposal (A, D).** Run A4 and the LLM guardrail runs on official free tiers (Ollama, Zhipu GLM-4-Flash, SiliconFlow, Gemini) through one OpenAI-compatible narrator. See PLAN.md §5.3. Team decision needed.

## Next steps by member

| Member | Now (W1-W3) |
|---|---|
| **A, lead** | Repo, CI, board, AI-use log; adversarial guardrail tests; table the two decisions above |
| **B, data and detection** | Explainer + 2 tests; investigate the precision gate and recommend; STL plots, STL vs MSTL rerun |
| **C, RCA** | Explainer + 2 tests; build and measure the top-3 fix; M0 calibration, brute-force agreement |
| **D, retrieval and evaluation** | Explainer + 2 tests; reproduce the dev results; prepare A3 distractor runs |

## Decisions log

| Date | Decision |
|---|---|
| 2026-10-15 | Free product: template narrator is the default; no paid API in runs, tests, demos or evaluation |
| 2026-10-19 | A4 with the paid API is out of scope; free-model run proposed, pending team decision |
