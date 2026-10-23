# Live demo script

Commands and the output to expect, on Windows PowerShell from the repo root after the README setup (`uv sync --extra dev --extra embeddings`). Everything here is free: the memo narrator is the template; no API key and no network are needed after the first MiniLM download.

Numbers below are from the 2026-10-23 runs on commit `c66ed09` (seed 42, default config). Timings depend on the machine.

## 1. Tests (about 3 minutes)

```powershell
.venv\Scripts\python -m pytest
```

Expect `315 passed`. For a short live check of the guardrails only:

```powershell
.venv\Scripts\python -m pytest -k "guardrail" -v
```

The test to point at: `test_my_reporting_test_an_invented_number_ends_in_the_template_memo` (tests/test_llm.py). An LLM draft that invents a number is rejected, retried with the violation as feedback, and then replaced by the template memo.

## 2. Generate the scenario (about 1 minute)

```powershell
.venv\Scripts\python -m kpi_rca generate --config config/default.yaml --out data
```

Writes `data/transactions.parquet` (1,765,633 checkout attempts over 21 days), `data/tickets.parquet` (798 tickets) and `data/ground_truth.json`, which holds the two injected incidents and one benign event:

| Event | Segment | Time (UTC) | True loss |
|---|---|---|---|
| INC-001 | country=DE, os=iOS, payment_gateway=PayPal | 09-10 13:00-17:00 | 602 orders, USD 38,034 |
| INC-002 | os=Android, app_version=5.4.0 | 09-16 17:00-20:00 | 738 orders, USD 36,360 |
| SURGE-US | country=US, traffic x1.8, conversion unchanged | 09-12 18:00-09-13 00:00 | none (must not be a kpi_drop) |

## 3. Run the pipeline (about 40 seconds)

```powershell
.venv\Scripts\python -m kpi_rca run --data data --out outputs
```

Expected window table:

| Window | Type | Time (UTC) | Result |
|---|---|---|---|
| W-001 | kpi_drop | 09-10 13:00-17:00 | memo INC-20260910-1300, top segment `country=DE & os=iOS & payment_gateway=PayPal` |
| W-002 | volume_surge | 09-12 18:00-09-13 02:00 | no memo (the surge is not called a drop) |
| W-003 | kpi_drop | 09-16 17:00-21:00 | memo INC-20260916-1700, top segment `os=Android & app_version=5.4.0` |

Open `outputs/reports/INC-20260910-1300.md` (the INC-001 memo, shown in full below) and `INC-20260910-1300_kpi.png` (conversion against its expected band). `INC-20260910-1300.json` holds the provenance: narrator, attempts, guardrail result, stage timings.

## 4. Streaming replay (about 75 seconds)

```powershell
.venv\Scripts\python -m kpi_rca stream --data data
```

Ends with:

```text
replayed 336 buckets in 73.1 s; 14 window(s)
  INC-001: detected by S-003 at 2026-09-10 15:00, delay 2.0 h
  INC-002: detected by S-010 at 2026-09-16 19:00, delay 2.0 h
```

Say plainly: the streaming replay also raises 3 kpi_drop alerts that match no incident (S-008, S-009, S-014). This is the detection-precision miss (VERIFICATION.md, criterion 2).

## 5. Optional: a free local LLM narrator

Only if Ollama is installed and `ollama pull gemma3:1b` has been run. It is never needed for the demo.

```powershell
.venv\Scripts\python -m kpi_rca run --data data --out outputs/run_ollama --narrator openai_compatible
```

On 2026-10-23 the INC-002 memo passed the guardrails on the first attempt, while the INC-001 drafts failed all 3 attempts. That memo fell back to the template, and `provenance.narrator` in its JSON says `template_fallback`. This is the point to show: an unchecked LLM sentence never reaches the memo.

## 6. Results to show (no need to rerun live)

- `outputs/evaluation/summary.md`: the 20-scenario benchmark (`evaluate --scenarios 20 --out outputs`, about 14 minutes without `--with-llm`, so it is not run live).
- `outputs/rees46_run_injected/reports/INC-20191025-1800.md` shows the pipeline on real REES46 traffic with one injected incident (VERIFICATION.md, Phase 3). It needs the Kaggle file, so show it from disk.

## INC-001 memo (expected output of step 3)

`outputs/reports/INC-20260910-1300.md`, template narrator:

```markdown
# INC-20260910-1300: SEV1: checkout conversion down 3.7% for DE / iOS / PayPal

Window W-001 (kpi_drop): 2026-09-10 13:00 to 2026-09-10 17:00 UTC. Severity SEV1, confidence high.

## Executive summary

Checkout conversion came in at 83.5% against 86.8% expected (down 3.7%) from 2026-09-10 13:00 to 2026-09-10 17:00 UTC. Estimated impact: USD 33,018 in lost revenue and 589 lost orders (USD 8,254 per hour), severity SEV1.

## Financial impact

| Metric | Value |
|---|---|
| Lost revenue | USD 33,018 |
| Lost orders | 589 |
| Revenue at risk per hour | USD 8,254 |
| Conversion (observed vs expected) | 83.5% vs 86.8% |
| Duration | 4 h |
| Severity | SEV1 |

## Root cause

The top RCA finding is the segment country DE, OS iOS and payment gateway PayPal. In the window, 88.7% of its attempts failed against 10.2% at baseline (8.67x). The segment had 761 attempts in the window and holds 105.4% of the excess failures. Top error code: PAYPAL_SDK_TIMEOUT (670 failures). Support received 39 similar tickets around the incident, 9.2x the usual rate.

| Rank | Segment | Failure rate (window) | Failure rate (baseline) | Share of excess failures | Attempts |
|---|---|---|---|---|---|
| 1 | country=DE & os=iOS & payment_gateway=PayPal | 88.7% | 10.2% | 105.4% | 761 |

## Customer evidence

39 similar tickets between 2026-09-10 11:00 and 2026-09-10 23:00 UTC, 9.2x the baseline rate (retrieval backend: sentence_transformers).

> Help! PayPal checkout keeps timing out in the iOS app. The PayPal window loads and then nothing happens. Order #355542.

> Hello, PayPal checkout keeps timing out in the iOS app. The PayPal window loads and then nothing happens. Please advise.

> Not happy. PayPal checkout keeps timing out in the iPhone app. The PayPal window loads and then nothing happens. What should I do?

## Recommended actions

| Priority | Action | Owner | Rationale |
|---|---|---|---|
| P0 | Contain the DE/iOS/PayPal failures: roll back the latest change or reroute this traffic. | Payments | 88.7% of the segment's attempts fail against 10.2% at baseline. |
| P1 | Alert on PAYPAL_SDK_TIMEOUT rates per segment so a repeat is caught within minutes. | SRE | PAYPAL_SDK_TIMEOUT is the most frequent error in the segment. |
| P1 | Send affected customers a status update and a retry link. | Customer Support | Customers are already contacting support about this incident. |

## Provenance

- Narrator: template (model: n/a)
- Guardrails: passed (1 narrator attempt(s))
- Stage timings: load_s 1.11 s, detect_s 0.50 s, index_s 16.79 s, rca_s 10.66 s, retrieval_s 0.05 s, impact_s 0.00 s, context_s 0.00 s, narrate_s 0.01 s
- Generated at: 2026-10-23 17:39 UTC
```

Questions to expect:

- **Why USD 33,018 when the truth is USD 38,034?** The estimate multiplies the lost orders (589 against 602 true, −2%) by the window's average order value. The true loss uses the actual baskets of the failed orders, so revenue is 13% low. On INC-002 it is 6% high.
- **What does "105.4% of the excess failures" mean?** The segment holds more failures than the whole window's excess over the baseline, because the rest of the traffic did slightly better than expected. More than 100% is possible and correct.
