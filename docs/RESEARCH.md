# RESEARCH.md: Literature review and gap analysis

| | |
|---|---|
| Project | Autonomous KPI Root-Cause Analysis and Executive Decision Engine via Multi-Modal ML and LLM Orchestration |
| Companion | `docs/PLAN.md` (design and evaluation), `src/kpi_rca/` (code) |
| Citation rule | Every entry in the reference list was checked against a publisher, proceedings, arXiv or author page. Where a detail (author order, DOI) could not be confirmed it is marked **[unverified]** or left out. No page numbers or DOIs were guessed. |

## 1. Introduction and scope

Dashboards can tell an analyst that checkout conversion fell. They do not say which customers were affected, whether customers are complaining, or what to do next. The project builds a pipeline with four parts:

1. **Detect.** Find anomalous hours in conversion, revenue and attempts (`detection/`).
2. **Localise.** Find the attribute=value segment that explains the excess failures (`rca/`).
3. **Retrieve.** Find the customer tickets that describe the same failure (`retrieval/`).
4. **Report.** Write an executive memo whose numbers are computed deterministically. An LLM writes only the narrative, under schema and grounding checks (`reporting/`).

This document surveys prior work for each stage and tests the three research gaps claimed in PLAN §1:

- (a) quantitative signals and qualitative feedback are analysed in silos;
- (b) finding the responsible attribute combination needs brute-force slicing;
- (c) LLMs miscalculate when given raw tables.

The aim is to defend the claims and, where the literature already covers them, to narrow them honestly. Every section ends by relating the literature to what is actually in `src/kpi_rca/`.

**Scope limits.**

- The review covers peer-reviewed papers and widely cited preprints up to 2025. It is not a systematic review: the searches covered arXiv, ACL Anthology, PVLDB, ACM DL and IEEE Xplore landing pages.
- The figure "2-6 h of manual SQL slicing and ticket reading" in the abstract has **no source found**. It is labelled an **assumption** here, as PLAN §1 asks. The closest evidence is Ghosh et al. (2022). They studied high-severity incidents at a large cloud service and found process gaps that delay detection, root-causing and mitigation, but they give no figure that transfers to business-KPI analysis.

## 2. Related work

### 2.1 KPI and time-series anomaly detection

**Prior work.**

- *Decomposition.* STL (Cleveland et al., 1990) splits a series into trend, seasonal and remainder with loess. Its robustness iterations down-weight outliers, so a multi-hour incident is not absorbed into the trend. RobustSTL (Wen et al., 2019) is a different algorithm with the same goal: it uses L1-based trend extraction and bilateral filtering to cope with abrupt trend changes. MSTL (Bandara et al., 2021/2025) extends STL to several seasonal periods, such as daily and weekly.
- *Robust statistics.* Twitter's S-H-ESD (Hochenbaum et al., 2017) is the closest industrial precedent. It removes an STL seasonal component, replaces the trend with the median and tests the residuals with a generalised ESD built on median and MAD. Leys et al. (2013) argue for MAD over the standard deviation when outliers are present.
- *Isolation-based detection.* Isolation Forest (Liu et al., 2008) scores points by how quickly random axis-parallel splits isolate them. It needs no labels and no density estimate.
- *Forecasting.* Prophet (Taylor & Letham, 2018) fits an additive model with trend, seasonality and holidays. Anomalies are points outside its forecast interval.
- *Deep and learned detectors.* SR-CNN at Microsoft (Ren et al., 2019) applies a spectral-residual saliency transform before a CNN. Donut (Xu et al., 2018) trains a VAE on seasonal web KPIs. Hundman et al. (2018) predict telemetry with LSTMs and set nonparametric dynamic thresholds on the errors.

**Limits.**

- Deep detectors need long histories and per-KPI tuning, and their scores are hard to explain to a business reader.
- Evaluation of the whole field is contested. Wu and Keogh (2021) show that several popular benchmarks are trivially solvable or mislabelled. Kim et al. (2022) show that the common *point-adjust* protocol, which counts a whole anomalous segment as detected if any point in it is flagged, inflates F1 so much that random scores can look competitive.

**Relation to the project.**

- `detector.py` uses statsmodels STL with `robust=True`: Cleveland's robustness weights, *not* Wen et al.'s RobustSTL. The settings are period 24 and seasonal smoother 21, with log1p for revenue and attempts.
- MSTL(24, 168) is used only after six weeks of history. The comment next to `uses_mstl` in `detector.py` records the comparison behind this rule: at three weeks MSTL produced more than 10x as many false |z| > 4 residuals as STL, while at six weeks the two were comparable. This is consistent with MSTL's need for several cycles of each period.
- Residuals are divided by their sampling noise (binomial for conversion, Poisson-like for counts) and converted to robust z with 1.4826·MAD. This follows the same idea as S-H-ESD, so the design is conventional, not novel.
- Isolation Forest (`contamination="auto"`) then combines the three z-series and their trailing means. This replaces hand-set thresholds, and Hochenbaum et al. is the precedent to cite when defending the choice.
- PLAN §4.1 calls Prophet, SARIMA and autoencoders "heavier and harder to explain". The literature supports this for Donut-, LSTM- and SR-CNN-style detectors at our data scale. It is a judgement about cost and explainability, not an accuracy result, and the report should present it that way.

### 2.2 Multi-dimensional root-cause localisation

**Prior work: AIOps localisation.** When a total KPI moves, the task is to find the attribute=value combinations, such as `country=BR ∧ os=iOS`, responsible for the change. Searching every combination of attributes and values grows exponentially, so the AIOps literature is mostly about avoiding that search.

- *Adtributor* (Bhagwan et al., 2014) ranks single-dimension elements by explanatory power and surprise (a Jensen-Shannon divergence between expected and observed share), with a succinctness preference. It assumes the root cause lies in one dimension.
- *iDice* (Lin et al., 2016) finds "effective combinations" of attributes, but in *customer issue reports*, not in a metric. It uses impact-based pruning, change detection and an isolation-power criterion.
- *HotSpot* (Sun et al., 2018) handles additive KPIs. It searches the cuboid lattice with Monte Carlo tree search, guided by a *ripple effect* assumption: a root-cause change spreads to descendant elements in proportion to their forecast values.
- *Squeeze* (Li et al., 2019) groups leaf elements by deviation score (bottom-up) and then searches each group (top-down). It is "generic" because it covers derived measures, such as success rate, as well as additive ones.
- *PSqueeze* (Li et al., 2023) extends Squeeze. It is evaluated on 5,400 injected faults, and its datasets are public.
- *AutoRoot* (Jing et al., 2021) replaces Squeeze's clustering with kernel density estimation.
- *RiskLoc* (Kalander, 2022) weights elements by a risk score. It reports element-level F1 on new synthetic datasets and states explicitly that brute force is infeasible beyond small cubes.

**Prior work: explanation in data management.** A parallel line of work treats the same problem as explaining data.

- *Discovery-driven OLAP* (Sarawagi et al., 1998) precomputes "surprise" values over a cube to guide drill-down.
- *Scorpion* (Wu & Madden, 2013) searches predicates that explain outlying aggregates.
- *Explanation tables* (El Gebaly et al., 2014) summarise a binary outcome with a small set of informative patterns.
- *MacroBase* (Bailis et al., 2017) mines attribute combinations with high relative risk, an itemset-mining method.
- *DIFF* (Abuzaid et al., 2018) generalises these explanation engines as one relational operator, with support and risk-ratio thresholds.
- *Slice Finder* (Chung et al., 2019) and *SliceLine* (Sagadeeva & Boehm, 2021) find data slices where a model performs badly. SliceLine prunes an exact enumeration with score upper bounds.
- *QuickInsights* (Ding et al., 2019) mines "insights" from multi-dimensional data at product scale.

**Limits.**

- Most AIOps methods assume the leaf-level forecast is accurate and that the root cause is a *deviation from forecast*. Few model baseline heterogeneity, for example a segment that is always worse. Adtributor's surprise and MacroBase's risk ratio both compare against a reference, but not against a model of each row's normal failure risk.
- Benchmarks are mainly synthetic injections into real or simulated cubes, as PSqueeze and RiskLoc show. This is the same weakness our evaluation has (§5).

**Relation to the project.** `rca/segment_model.py`, `rca/search.py` and `rca/baseline.py` implement the following:

- **Excess-failure framing.** M0 is a LightGBM model of each attempt's baseline failure probability p0, trained on seven baseline days. The window's excess for row i is y_i − p0_i. A slice is scored by EP, the share of the window's total excess inside it (close to recall), and purity, the excess divided by the slice's expected successes (close to precision). The score is their harmonic mean.
- **Significance.** A one-sided two-proportion z-test compares the slice with its own baseline rate, using a Bonferroni-corrected alpha. Without this test, the always-worse BR segment would be blamed (the "BR trap", ablation A2).
- **Candidate generation.** Candidates come from M1, a model fitted with `init_score = logit(p0)` so that it learns only the change. For the failed window rows that M1 pushes up most (largest change margin), the features with clearly positive TreeSHAP values are kept, frequent subsets of them up to four attributes form the candidates, and the best eligible slice is chosen, pruned and removed. The process repeats for up to three findings.
- **Comparator.** `baseline.py` enumerates every slice up to three attributes with the same score, test and selection rule, and reports `slices_evaluated` and runtime. It has no prune step, and its Bonferroni divisor is the number of slices it scores per round (hundreds, against at most 25 mined candidates for the guided search). On the default data it scores about 1,800-1,900 slices in 3-5 s, while the guided search scores about 50 in 10-12 s: the saving is in evaluations, not wall time.

EP is close to Adtributor's explanatory power, the purity and score pairing resembles a precision and recall F-measure, and the iterative "remove and repeat" step resembles Squeeze's per-cluster search and MacroBase's ranking. These pieces are borrowed. What we found no precedent for is the combination of (i) an offset model that turns the problem into localising *excess over each row's own baseline*, and (ii) SHAP attributions as the candidate generator.

### 2.3 Gradient-boosted trees and SHAP

**Prior work.**

- LightGBM (Ke et al., 2017) speeds up gradient boosting with gradient-based one-side sampling and exclusive feature bundling.
- SHAP (Lundberg & Lee, 2017) unifies additive attribution methods under Shapley values.
- TreeSHAP (Lundberg et al., 2018) computes exact SHAP values for tree ensembles in polynomial time and defines SHAP *interaction values*. Lundberg et al. (2020) extend this to global explanations. LightGBM exposes the same algorithm through `pred_contrib`.

**Limits.** Kumar et al. (2020) show that Shapley-based importance depends on how absent features are modelled, can mislead under correlated features, and is not a causal statement. In our setting a SHAP value says "M1 raised this row's predicted failure because of `os=iOS`". It does not say "iOS caused the incident".

**Relation to the project.**

- SHAP is used only to *propose* candidates. Every candidate is then re-scored on raw counts against p0 and must pass the significance test, so a misleading attribution can cost recall (a missed candidate) but cannot by itself create a finding.
- This is the honest form of the claim. The A3 comparison with brute force, where both use the same score, measures exactly how much recall the SHAP proposals lose.
- Nested attributes, such as an app version that implies an OS, are a known case where correlated features split SHAP credit, and PLAN lists them as a threat to validity.
- We found no published method that uses SHAP values to generate attribute-combination candidates for multi-dimensional KPI localisation. Single-feature SHAP rankings appear in patents and blog posts, but these are not peer-reviewed. This is a statement about our search, not proof that no such work exists.

### 2.4 Dense retrieval for tickets

**Prior work.**

- Sentence-BERT (Reimers & Gurevych, 2019) trains siamese BERT encoders so that cosine similarity between sentence embeddings is meaningful.
- MiniLM (Wang et al., 2020) distils transformers into small students. `all-MiniLM-L6-v2` is a sentence-transformers checkpoint built on it; the model card is the source, not a paper.
- FAISS (Johnson et al., 2021; Douze et al., 2024) provides exact and approximate vector search.
- LSA (Deerwester et al., 1990) is the classic low-rank lexical baseline.
- BEIR (Thakur et al., 2021) found that BM25 is a strong zero-shot baseline and that dense retrievers often fail to beat it out of domain.

**Limits.** Dense retrievers are sensitive to domain shift, and short, templated, jargon-heavy tickets are exactly where lexical methods do well.

**Relation to the project.**

- `retrieval/` embeds tickets with MiniLM (384-d) or LSA (TF-IDF plus truncated SVD, 128-d) and stores unit vectors in an exact `IndexFlatIP`, so the inner product equals cosine similarity.
- Retrieval is filtered to the window from 2 h before to 6 h after, and boosted by 0.05 each for a matching country and OS.
- The query is built from the localised segment and its error codes.
- Ablation A1 (MiniLM vs LSA) matches the literature's main comparison. Following BEIR, a BM25 arm would be a cheap and more defensible lexical baseline than LSA. This is optional (§4).

### 2.5 LLMs for incident management

**Prior work.**

- Ahmed et al. (2023) fine-tuned and prompted GPT-3.x models on more than 40,000 Microsoft incidents to recommend root causes and mitigations. Their input was incident title and summary text.
- RCACopilot (Chen et al., 2024) first runs handlers that collect multi-source diagnostics, then asks an LLM to predict a root-cause category with an explanation.
- Zhang et al. (2024) use in-context learning with GPT-4 for root-causing. Oasis (Jin et al., 2023) summarises outage impact.
- Agentic, tool-using systems followed: a ReAct agent for RCA (Roy et al., 2024) and RCAgent (Wang et al., 2024).
- OpenRCA (Xu et al., 2025) tests LLMs on 335 failures with 68 GB of telemetry. The best configuration solved about 11% of cases.

**Limits.**

- These systems target *software incidents*, using logs, traces and tickets, rather than business KPIs.
- They mostly evaluate category accuracy or similarity to human root-cause text, not the factual accuracy of every number in the output.
- OpenRCA shows that LLMs given raw telemetry, even with tools, are far from reliable.

**Relation to the project.**

- The project is closest in spirit to RCACopilot and RCAgent: deterministic collectors gather the evidence, and the LLM reasons over a curated summary.
- The difference is that our LLM is not asked to *find* the cause. Detection, localisation, impact and retrieval are all done before the LLM call, and the LLM only narrates.
- This is a deliberate reduction of the LLM's role that OpenRCA's results support.

### 2.6 LLM numeracy, tables, hallucination, tool use and structured output

**Prior work.**

- *Hallucination.* Ji et al. (2023) and Huang et al. (2025) survey hallucination. Unfaithful numbers and entities are a central category in both.
- *Data-to-text.* Wiseman et al. (2017) showed that generated reports are often fluent but wrong about the records. They measured fidelity extractively by checking each stated fact against the source table, which is the same idea as our numeric-grounding guardrail.
- *Tables and long inputs.* Sui et al. (2024) find that LLM table understanding depends strongly on serialisation format. Liu et al. (2024) show that information in the middle of long contexts is used poorly, which matters when a large table is pasted into a prompt.
- *Arithmetic.* GSM-Symbolic (Mirzadeh et al., 2025) shows that accuracy drops when only the numbers in a problem change, or when an irrelevant clause is added.
- *Offloading computation.* The remedy in the literature is to move the computation out of the model. PAL (Gao et al., 2023) and Program of Thoughts (Chen et al., 2023) have the LLM write code that an interpreter executes. Toolformer (Schick et al., 2023) learns calculator and API calls. InsightPilot (Ma et al., 2023) has an LLM direct a deterministic insight engine (QuickInsights and others) over multi-dimensional data.
- *Structured output.* Constrained decoding (Willard & Louf, 2023) guarantees schema-valid output. Tam et al. (2024) find that strict format constraints can *reduce* reasoning quality, so a valid schema does not mean the content is true.
- *Prompt injection.* Greshake et al. (2023) show indirect prompt injection through retrieved content. This is relevant because customer tickets are untrusted text placed into the prompt.

**Relation to the project.**

- `reporting/context.py` builds a compact JSON of precomputed facts. `reporting/llm.py` calls Claude with a Pydantic output schema (`messages.parse`, `output_format=MemoNarrative`), and stops other than `end_turn` are treated as errors.
- `reporting/guardrails.py` checks the output:
  - schema and length;
  - every number against the facts, within 2%;
  - consistency of the named segment;
  - that quotes are verbatim substrings of retrieved tickets.
- Failures are retried up to twice and then replaced by a deterministic template narrator.
- This applies the PAL and tool-use principle ("compute outside, narrate inside") and adds Wiseman-style extractive verification on top of schema enforcement. That addition answers Tam et al.'s point that schema-valid is not the same as true.

## 3. Gap analysis

### 3.1 Gap (a): quantitative/qualitative silo

**For.**

- Mainstream KPI localisation (Adtributor, HotSpot, Squeeze, RiskLoc) uses only the metric cube. Mainstream incident LLM work (Ahmed et al., Zhang et al.) mostly uses only incident text.
- None of the localisation papers retrieve customer feedback for the localised segment.

**Against.**

- Fusion already exists:
  - iDice mines attribute combinations *from customer issue reports*;
  - iFeedback (2019) detects issues in real time from user feedback;
  - Jiang et al. (2025) study feedback-based issue detection on a service with a billion users;
  - Eadro (Lee et al., 2023) and Nezha (Yu et al., 2023) fuse metrics, logs and traces;
  - RCACopilot and Oasis aggregate multi-source diagnostics before calling an LLM.
- Jiang et al. also find that much feedback is irrelevant, and that some severe issues cannot be detected from feedback alone. This argues against assuming tickets are always informative.

**Verdict: partially supported, so narrow the claim.** The claim "nobody combines numbers and text" is false. The defensible claim is narrower: for a *business KPI*, the project uses the *output of segment localisation* (attribute values plus error codes) and the anomaly window to drive semantic ticket retrieval, then presents both in one memo with the evidence attributed. A1, P@k and spike_ratio test whether that coupling works. Because ticket text is templated and generated alongside the incidents, it does not test whether real feedback is informative (§5).

### 3.2 Gap (b): combinatorial isolation without brute force

**For.** Brute-force enumeration grows quickly: 1,255 slices with the default configuration and 124,600 at 10 attributes × 10 values (PLAN §4.3). RiskLoc and HotSpot also describe exhaustive search as infeasible at scale.

**Against.** Avoiding brute force is *the* central topic of multi-dimensional RCA and data-explanation research: HotSpot's MCTS, Squeeze's clustering, RiskLoc's risk weighting, iDice's pruning, MacroBase's and DIFF's itemset mining, and SliceLine's bounded enumeration all do it. Squeeze already covers derived ratio measures, so "non-additive KPI" is not new either.

**Verdict: largely addressed by prior work, so the novelty is narrow.** The defensible contributions are:

1. the excess-over-baseline framing with an M0 offset and a per-segment baseline z-test, which removes the always-worse-segment confound (A2 measures this directly);
2. SHAP-guided candidate generation. We found no peer-reviewed precedent, and A3 measures its cost and recall against brute force under the same score.

Comparing against brute force alone shows efficiency, not competitiveness with the state of the art. At least one multi-dimensional literature baseline is recommended (§4).

### 3.3 Gap (c): LLM math on raw tables

**For.** The problem is well documented:

- unfaithful data-to-text generation (Wiseman et al.);
- sensitivity to table format (Sui et al.);
- loss of information in long contexts (Liu et al.);
- fragile arithmetic (Mirzadeh et al.);
- low RCA accuracy on raw telemetry (OpenRCA);
- the hallucination surveys.

A table with about 1.8M rows does not fit in a context window at all, so the raw approach is not an option at our scale.

**Against.** The solution pattern is established: PAL, PoT, Toolformer, RCAgent and InsightPilot all compute outside the model. Structured outputs are now a standard API feature.

**Verdict: the problem is strongly supported, but the solution is not new.** The contribution is the *engineering and measurement*: a compact facts context, deterministic numeric, segment and quote grounding checks, bounded retries, a template fallback that always passes, and the A4 ablation that measures the hallucination rate with and without the context. Presented as "an empirically checked application of compute-then-narrate", it is defensible. Presented as "a new solution to LLM numeracy", it is not.

## 4. Evaluation mapping to the literature

The PLAN §5 metrics map onto the literature as follows.

| Stage | PLAN §5 metric | Literature practice | Comment |
|---|---|---|---|
| Detection | Window-overlap P/R/F1, delay, false alarms per day | Point-adjust F1 (Xu et al., 2018) criticised by Kim et al. (2022); UCR archive (Wu & Keogh, 2021) | Overlap-based F1 is event-level and close to point-adjust. PLAN reports delay and false alarms from an incident-free run alongside it, which addresses Kim et al.'s criticism. Keep all three together. |
| RCA | Exact match, Jaccard, top-3, slices_evaluated, runtime | Element-level F1 = 2TP/(2TP+FP+FN) (Squeeze; PSqueeze; Kalander, 2022) | Exact match corresponds to their per-case hit, and Jaccard to overlap. Also reporting element-level F1 would make the numbers comparable with published tables. |
| RCA baseline | Brute force; optional Adtributor in W8 | Squeeze/PSqueeze and RiskLoc have public code and data | Adtributor is single-dimension (PLAN §4.3 is correct) and will lose on the three-way INC-001 by design. A multi-dimensional baseline (RiskLoc or Squeeze) is a fairer opponent. |
| Retrieval | P@5, P@10, R@10 (R@10 capped near 0.37 for INC-001) | P@k and recall@k, as in BEIR (Thakur et al., 2021) | Report the cap with the score. A BM25 arm is optional. |
| Report | Guardrail pass rates; A4 hallucination rate | Extractive fact checking against records (Wiseman et al., 2017) | A4 scores outputs deterministically against the fact set with the guardrail tolerance. This avoids LLM-as-judge bias. |
| Protocol | 20 scenarios, held-out seed, 95% bootstrap CI | Synthetic injection is standard in RCA (PSqueeze, RiskLoc) | A frozen config plus a held-out seed is the right guard against tuning to the test set. |

**Later validation (PLAN §3.3).**

- The Squeeze/PSqueeze datasets (NetManAIOps GitHub, Zenodo) and the RiskLoc synthetic sets are public multi-dimensional RCA benchmarks.
- The UCR anomaly archive (Wu & Keogh) is a better detection benchmark than older ones such as Yahoo/NAB.
- For a real complaint corpus, public consumer-complaint databases exist (for example the US CFPB database). It has not been reviewed here **[unverified suitability]**.

## 5. Threats to validity

1. **Synthetic data (the main threat).**
   - The generator, the injected incidents, the ticket templates and the evaluator were all written by the same team. Good scores may only show that the pipeline inverts its own generator. PLAN calls this the circularity threat.
   - The literature has the same weakness, since PSqueeze and RiskLoc also rely on injection. That makes the threat common, not acceptable.
   - Mitigations: held-out seeds, distractor attributes, benign surges, the BR confound, and validation on the public benchmarks in §4.
2. **Templated tickets.** Templated text favours lexical matching and makes retrieval easy. Real feedback is noisier and often irrelevant (Jiang et al., 2025). A1's MiniLM-vs-LSA gap will probably understate real-world differences.
3. **Correlated and nested attributes.** These split SHAP credit (Kumar et al., 2020) and can make the guided search miss a candidate that brute force finds. A3 with nested distractors should be reported separately.
4. **Detection evaluation.** Overlap-based metrics inflate scores (Kim et al., 2022). Batch STL is two-sided, so batch detection is retrospective; only `streaming.py` replay gives honest delay numbers.
5. **Multiple testing.** Many slices are tested per window. The code applies a Bonferroni correction, which is conservative and can lower recall for small segments. That trade-off should be stated.
6. **Small scale.** Twenty scenarios and ten A4 scenarios × 3 repeats give wide confidence intervals. Results should be read as effect directions with CIs, not leaderboard numbers.
7. **LLM non-determinism and drift.** Model versions change. A4 and the guardrail pass rates hold only for the model ID and date recorded with the run.
8. **The "2-6 h" baseline is an assumption** (§1). Time-to-diagnosis can only be compared against it, not against measured analyst time.

## 6. References

Author-year style. "[unverified]" marks details that could not be confirmed.

- Abuzaid, F., et al. (2018). DIFF: A Relational Interface for Large-Scale Data Explanation. *PVLDB* 12(4). https://doi.org/10.14778/3297753.3297761
- Ahmed, T., Ghosh, S., Bansal, C., Zimmermann, T., Zhang, X., Rajmohan, S. (2023). Recommending Root-Cause and Mitigation Steps for Cloud Incidents using Large Language Models. *ICSE 2023*. https://arxiv.org/abs/2301.03797
- Bailis, P., Gan, E., Madden, S., Narayanan, D., Rong, K., Suri, S. (2017). MacroBase: Prioritizing Attention in Fast Data. *SIGMOD 2017*. https://doi.org/10.1145/3035918.3035928
- Bandara, K., Hyndman, R.J., Bergmeir, C. (2025). MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns. *Int. J. Operational Research* 52(1):79-98. Preprint 2021: https://arxiv.org/abs/2107.13462
- Bhagwan, R., et al. (2014). Adtributor: Revenue Debugging in Advertising Systems. *NSDI 2014*, 43-55.
- Chen, W., Ma, X., Wang, X., Cohen, W.W. (2023). Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. *TMLR*. https://arxiv.org/abs/2211.12588
- Chen, Y., et al. (2024). Automatic Root Cause Analysis via Large Language Models for Cloud Incidents (RCACopilot). *EuroSys 2024*. https://arxiv.org/abs/2305.15778
- Chung, Y., Kraska, T., Polyzotis, N., Tae, K.H., Whang, S.E. (2019). Slice Finder: Automated Data Slicing for Model Validation. *ICDE 2019*. https://arxiv.org/abs/1807.06068
- Cleveland, R.B., Cleveland, W.S., McRae, J.E., Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics* 6(1):3-73.
- Deerwester, S., Dumais, S.T., Furnas, G.W., Landauer, T.K., Harshman, R. (1990). Indexing by Latent Semantic Analysis. *JASIS* 41(6):391-407.
- Ding, R., Han, S., Xu, Y., Zhang, H., Zhang, D. (2019). QuickInsights: Quick and Automatic Discovery of Insights from Multi-Dimensional Data. *SIGMOD 2019*. https://doi.org/10.1145/3299869.3314037
- Douze, M., et al. (2024). The Faiss library. https://arxiv.org/abs/2401.08281
- El Gebaly, K., Agrawal, P., Golab, L., Korn, F., Srivastava, D. (2014). Interpretable and Informative Explanations of Outcomes. *PVLDB* 8(1):61-72. https://doi.org/10.14778/2735461.2735467
- Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., Neubig, G. (2023). PAL: Program-aided Language Models. *ICML 2023*, PMLR 202:10764-10799. https://arxiv.org/abs/2211.10435
- Ghosh, S., et al. (2022). How to Fight Production Incidents? An Empirical Study on a Large-scale Cloud Service. *SoCC 2022*. https://doi.org/10.1145/3542929.3563482
- Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., Fritz, M. (2023). Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. *AISec 2023*, 79-90. https://arxiv.org/abs/2302.12173
- Hochenbaum, J., Vallis, O.S., Kejariwal, A. (2017). Automatic Anomaly Detection in the Cloud Via Statistical Learning. https://arxiv.org/abs/1704.07706
- Huang, L., et al. (2025). A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. *ACM TOIS*. https://doi.org/10.1145/3703155
- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T. (2018). Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. *KDD 2018*. https://arxiv.org/abs/1802.04431
- iFeedback [authors unverified] (2019). iFeedback: Exploiting User Feedback for Real-time Issue Detection in Large-Scale Online Service Systems. *ASE 2019*, 352-363.
- Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys* 55(12). https://arxiv.org/abs/2202.03629
- Jiang, S., Gu, J., Zheng, W., Zhou, Y., Lyu, M.R. (2025). Can User Feedback Help Issue Detection? An Empirical Study on a One-billion-user Online Service System. *ESEM 2025*. https://arxiv.org/abs/2508.00593
- Jin, P., Zhang, S., et al. (2023). Assess and Summarize: Improve Outage Understanding with Large Language Models. *ESEC/FSE 2023*, 1657-1668. https://doi.org/10.1145/3611643.3613891
- Jing, et al. [full author list unverified] (2021). AutoRoot: A Novel Fault Localization Schema of Multi-dimensional Root Causes. *IEEE WCNC 2021*, 1-7.
- Johnson, J., Douze, M., Jégou, H. (2021). Billion-scale Similarity Search with GPUs. *IEEE Transactions on Big Data* 7(3):535-547. https://arxiv.org/abs/1702.08734
- Kalander, M. (2022). RiskLoc: Localization of Multi-dimensional Root Causes by Weighted Risk. https://arxiv.org/abs/2205.10004
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*.
- Kim, S., Choi, K., Choi, H.-S., Lee, B., Yoon, S. (2022). Towards a Rigorous Evaluation of Time-Series Anomaly Detection. *AAAI 36*(7):7194-7201. https://doi.org/10.1609/aaai.v36i7.20680
- Kumar, I.E., Venkatasubramanian, S., Scheidegger, C., Friedler, S. (2020). Problems with Shapley-value-based Explanations as Feature Importance Measures. *ICML 2020*, PMLR 119:5491-5500.
- Lee, C., Yang, T., Chen, Z., Su, Y., Lyu, M.R. (2023). Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data. *ICSE 2023*, 1750-1762. https://arxiv.org/abs/2302.05092
- Leys, C., Ley, C., Klein, O., Bernard, P., Licata, L. (2013). Detecting Outliers: Do Not Use Standard Deviation around the Mean, Use Absolute Deviation around the Median. *J. Experimental Social Psychology* 49(4):764-766. https://doi.org/10.1016/j.jesp.2013.03.013
- Li, Z., et al. (2019). Generic and Robust Localization of Multi-Dimensional Root Causes (Squeeze). *ISSRE 2019*, 47-57. Code and data: https://github.com/NetManAIOps/Squeeze
- Li, Z., Chen, J., et al. (2023). Generic and Robust Root Cause Localization for Multi-Dimensional Data in Online Service Systems (PSqueeze). *Journal of Systems and Software*, 111748. https://doi.org/10.1016/j.jss.2023.111748
- Lin, Q., Lou, J.-G., Zhang, H., Zhang, D. (2016). iDice: Problem Identification for Emerging Issues. *ICSE 2016*, 214-224. https://doi.org/10.1145/2884781.2884795
- Liu, F.T., Ting, K.M., Zhou, Z.-H. (2008). Isolation Forest. *ICDM 2008*, 413-422. https://doi.org/10.1109/ICDM.2008.17
- Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL* 12:157-173. https://doi.org/10.1162/tacl_a_00638
- Lundberg, S.M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*. https://arxiv.org/abs/1705.07874
- Lundberg, S.M., Erion, G.G., Lee, S.-I. (2018). Consistent Individualized Feature Attribution for Tree Ensembles. https://arxiv.org/abs/1802.03888
- Lundberg, S.M., et al. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence* 2(1):56-67. https://doi.org/10.1038/s42256-019-0138-9
- Ma, P., Ding, R., Wang, S., Han, S., Zhang, D. (2023). InsightPilot: An LLM-Empowered Automated Data Exploration System. *EMNLP 2023 System Demonstrations*, 346-352. https://doi.org/10.18653/v1/2023.emnlp-demo.31
- Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Bengio, S., Farajtabar, M. (2025). GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. *ICLR 2025*. https://arxiv.org/abs/2410.05229
- Reimers, N., Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*. https://arxiv.org/abs/1908.10084
- Ren, H., et al. (2019). Time-Series Anomaly Detection Service at Microsoft. *KDD 2019*. https://doi.org/10.1145/3292500.3330680
- Roy, D., et al. (2024). Exploring LLM-based Agents for Root Cause Analysis. *FSE 2024 Companion*. https://arxiv.org/abs/2403.04123
- Sagadeeva, S., Boehm, M. (2021). SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging. *SIGMOD 2021*. https://doi.org/10.1145/3448016.3457323
- Sarawagi, S., Agrawal, R., Megiddo, N. (1998). Discovery-driven Exploration of OLAP Data Cubes. *EDBT 1998*, LNCS 1377:168-182. https://doi.org/10.1007/BFb0100984
- Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS 36*. https://arxiv.org/abs/2302.04761
- Sui, Y., Zhou, M., Zhou, M., Han, S., Zhang, D. (2024). Table Meets LLM: Can Large Language Models Understand Structured Table Data? *WSDM 2024*. https://arxiv.org/abs/2305.13062
- Sun, Y., Zhao, Y., Su, Y., Liu, D., Nie, X., Meng, Y., Cheng, S., Pei, D., Zhang, S., Qu, X., Guo, X. (2018). HotSpot: Anomaly Localization for Additive KPIs With Multi-Dimensional Attributes. *IEEE Access* 6:10909-10923. https://doi.org/10.1109/ACCESS.2018.2804764
- Tam, Z.R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H.-y., Chen, Y.-N. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models. *EMNLP 2024 Industry*. https://arxiv.org/abs/2408.02442
- Taylor, S.J., Letham, B. (2018). Forecasting at Scale. *The American Statistician* 72(1):37-45. https://doi.org/10.1080/00031305.2017.1380080
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS Datasets and Benchmarks*. https://arxiv.org/abs/2104.08663
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *NeurIPS 33*. https://arxiv.org/abs/2002.10957
- Wang, Z., et al. (2024). RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models. *CIKM 2024*. https://doi.org/10.1145/3627673.3680016
- Wen, Q., Gao, J., Song, X., Sun, L., Xu, H., Zhu, S. (2019). RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series. *AAAI 33*(01):5409-5416. https://doi.org/10.1609/aaai.v33i01.33015409
- Willard, B.T., Louf, R. (2023). Efficient Guided Generation for Large Language Models. https://arxiv.org/abs/2307.09702
- Wiseman, S., Shieber, S., Rush, A. (2017). Challenges in Data-to-Document Generation. *EMNLP 2017*, 2253-2263. https://doi.org/10.18653/v1/D17-1239
- Wu, E., Madden, S. (2013). Scorpion: Explaining Away Outliers in Aggregate Queries. *PVLDB* 6(8):553-564.
- Wu, R., Keogh, E. (2021). Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress. *IEEE TKDE* [volume and DOI unverified]. https://arxiv.org/abs/2009.13807
- Xu, H., et al. (2018). Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications (Donut). *WWW 2018*, 187-196. https://arxiv.org/abs/1802.03903
- Xu, J., et al. (2025). OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures? *ICLR 2025*.
- Yu, G., et al. [author order unverified] (2023). Nezha: Interpretable Fine-Grained Root Causes Analysis for Microservices on Multi-modal Observability Data. *ESEC/FSE 2023*. https://doi.org/10.1145/3611643.3616249
- Zhang, X., Ghosh, S., Bansal, C., Wang, R., Ma, M., Kang, Y., Rajmohan, S. (2024). Automated Root Causing of Cloud Incidents using In-Context Learning with GPT-4. *FSE 2024 Industry*. https://arxiv.org/abs/2401.13810
