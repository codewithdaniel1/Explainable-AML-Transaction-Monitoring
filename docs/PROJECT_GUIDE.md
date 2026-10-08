# Project guide: explainable AML modeling

The [notebook](../aml_modeling.ipynb) contains the complete runnable class-project MVP and its saved outputs. This guide explains the study design, each notebook step, the model comparison, and where Databricks and dbt would fit. Use the [README](../README.md) for installation and the IBM data download.

## Research question and scope

**Question:** Can interpretable models rank IBM's synthetic laundering-labeled transactions nearly as well as XGBoost while giving useful explanations at a fixed investigator review capacity?

The comparison is between logistic regression, XGBoost with Tree SHAP, and an Explainable Boosting Machine (EBM). Currency-aware rules are a simple operational baseline. The output is a ranked **review queue**; neither a model score nor a synthetic label is a real suspicious activity finding.

This is a **10% sampled modeling pilot**, not a full-data model benchmark. The sample is drawn from September 1–10 with seed 42. Behavioral features for each sampled transaction are calculated from **all earlier transactions in the original CSV**, so sampling does not remove its prior history. All 1,108 later rows are retained for a separate stress check.

## How the notebook flows

| Section | Step | What to check in the output |
| --- | --- | --- |
| **0** | Set the CSV path, sample fraction, random seed, and 100-alert review capacity. | The kernel and local CSV are found. |
| **1** | Scan and profile the entire IBM CSV in chunks while selecting modeling rows. | Row and label counts, daily volume, and the late-period jump in label rate. |
| **2** | Assign fixed calendar periods before training. | Training, validation, and primary holdout each have positive and negative labels. |
| **3** | Use DuckDB windows to calculate earlier sender activity from all source rows. | One feature row aligns with each sampled transaction. Temporary database files are removed after the cell. |
| **4–5** | Define and fit the rules and six ML specifications. | Each model's validation and holdout average precision; the chosen specification comes from validation only. |
| **6** | Plot average precision, precision–recall curves, and labels found as review capacity grows. | Compare ranking quality and detection at 100 alerts. |
| **7** | Inspect highest-scored alerts, missed positives, currency groups, and a descriptive bootstrap interval. | See which errors matter and which group estimates have few positives. |
| **8** | Inspect logistic coefficients, XGBoost importance and Tree SHAP, and EBM terms. | Distinguish a model explanation from evidence about a transaction. |
| **9** | Replay stored **in-memory** scores with a cutoff chosen from validation scores. | See how alert volume changes; the late period remains a separate stress check. |
| **10** | Change the review variables to inspect a ranked synthetic case and its earlier sender history. | Read the case context and model explanation in the notebook. |
| **11** | Read the limits before presenting the result. | Keep claims tied to the sampled, synthetic study. |

The raw CSV lives at `data/raw/HI-Small_Trans.csv`. The notebook holds its displayed results and does not create a `results/` folder, metrics JSON, or predictions CSV. A fresh **Run All** recalculates the experiment and needs temporary disk space for DuckDB.

## Data, time windows, and leakage control

The label is `is_laundering`. The full HI-Small CSV has **5,078,345 transactions** and **5,177 positive labels** (about **0.102%**). Most days before September 11 have hundreds of thousands of transactions; September 11–18 contains only **1,108 rows**, including **655 positives**. That abrupt change could be related to synthetic generation or truncation, but the dataset alone does not establish its cause.

The study uses fixed, chronological windows: **September 1–6 training**, **September 7–8 validation**, **September 9–10 primary holdout**, and **September 11–18 late stress**. Models fit on training labels only. Validation average precision selects the specification. The holdout is then used to report the selected model's performance. The late period is shown separately because its prevalence is so different; its AP cannot be compared directly with primary-holdout AP.

Current-transaction inputs are log amount paid, hour, day of week, same-bank indicator, payment format, payment currency, and receiving currency. Behavioral inputs add the sender's earlier 24-hour and seven-day counts, paid-amount history, amount relative to a prior average, and whether the counterparty is new in the past seven days. Account and bank IDs **construct** history but are not direct model columns.

The feature query includes **strictly earlier timestamps**. Transactions at the same timestamp cannot see one another, even if they appear earlier in CSV row order. Count history spans the sender's currencies, while amount sums and averages use the **same payment currency** as the transaction being scored. These choices prevent two common forms of artificial history or invalid currency mixing.

## Why these models

| Model | What it learns | How it is explained | Main tradeoff |
| --- | --- | --- | --- |
| **Currency-aware rules** | No fitted classifier. A transaction gets points for a training-only 99th-percentile amount cutoff in its payment currency, high prior 24-hour count, and an amount far above its own prior average. | The triggered conditions are directly visible. | Easy to inspect, but these illustrative thresholds may miss patterns. |
| **Logistic regression** | A weighted linear relationship after scaling numeric inputs and one-hot encoding categories. | Coefficients show how encoded inputs change the fitted log-odds score. | Simple and familiar, but it cannot learn complex interactions unless they are built into the inputs. |
| **XGBoost** | Weighted boosted trees; this run uses 200 trees, depth 4, and learning rate 0.05. | Feature importance summarizes the fit; native Tree SHAP gives contributions for selected transactions on the raw log-odds scale. | Flexible ranking, with explanations that require care to interpret. |
| **EBM** | Weighted nonlinear additive effects for named features. Interactions are disabled in this pilot. | Global term importance shows which learned feature effects matter most. | More transparent structure, but its additive setup may miss interactions captured by trees. |

Each ML algorithm runs with **transaction-only** and **transaction-plus-behavioral** inputs, giving six ML specifications plus the rules. All use the same training, validation, and primary-test calendar windows. Class weighting responds to rare labels; it also means the resulting scores should **not** be read as calibrated laundering probabilities.

## Metrics and reported result

**Average precision (AP)** summarizes precision across recall levels and is the model-selection metric. The notebook's internal key `pr_auc` calls scikit-learn's `average_precision_score`. At a fixed review capacity, **Precision@100** is positive labels divided by 100 reviewed transactions; **Recall@100** is positive labels found divided by all positives in that period. ROC-AUC is another ranking measure. Brier score is computed as a diagnostic, but the weighted outputs are not calibrated probabilities.

| Specification | Validation AP | Primary-holdout AP | Positive labels in top 100 |
| --- | ---: | ---: | ---: |
| Currency-aware rules | 0.0010 | 0.0011 | 0 |
| Logistic, transaction only | 0.0120 | 0.0135 | 1 |
| XGBoost, transaction only | 0.0336 | 0.0459 | 12 |
| EBM, transaction only | 0.0185 | 0.0250 | 1 |
| Logistic, behavioral | 0.0133 | 0.0111 | 0 |
| **XGBoost, behavioral** | **0.3362** | **0.2245** | **23** |
| EBM, behavioral | 0.1519 | 0.1105 | 18 |

Validation AP selects **behavioral XGBoost**. The sampled primary holdout has **86,066 rows and 103 positives**. Of its top 100 alerts, **23 are positive**: 23% precision and 22.3% recall at that capacity. The 300-repeat stratified **row** bootstrap gives a descriptive 95% AP interval of **0.1468–0.3128**. It does not account for repeat activity by the same accounts or time dependence. Behavioral inputs help XGBoost and EBM in this run; they do not improve every model (compare the logistic rows).

Explanations describe **what the fitted model used**, not why the source simulator assigned a label and not whether real laundering occurred. The case-review section is a demonstration on synthetic data. The replay uses precomputed scores and a validation-only cutoff; it is not live monitoring.

## Databricks and dbt: current status and a concrete later path

**Current status:** neither Databricks nor dbt runs in this repository. The notebook uses local DuckDB and pandas to keep the ML experiment reproducible without a cloud workspace. The original proposal listed **Databricks + dbt as a preferred engineering extension**, not as a separate ML research question. Calling the current notebook a Databricks/dbt pipeline would be inaccurate.

If cloud engineering becomes part of the next deliverable, the roles would be:

| Stage | Proposed responsibility | Equivalence check before using it for ML |
| --- | --- | --- |
| **Load** | Put the original IBM CSV into a Databricks raw table. Databricks supports loading CSV into Delta tables; dbt assumes data has already been loaded. | Match 5,078,345 source rows, 5,177 positives, schema, IDs, timestamps, and a stable source-row identifier. |
| **Transform with dbt** | Declare the raw table as a dbt source; create SQL staging/feature models for types, names, quality checks, and time-safe behavioral features. dbt models run SQL on the connected platform. | Check label values, missing fields, unique source-row IDs, same-time exclusion, currency-specific amount history, and feature parity against notebook examples. |
| **Model** | Feed the validated feature table into the notebook's unchanged train/validation/holdout comparison. | Keep the same dates, sample seed or row set, seven specifications, and metric definitions before comparing results. |

Databricks documents that [dbt transforms already-loaded data](https://docs.databricks.com/aws/en/partners/prep/dbt), and dbt documents [sources](https://docs.getdbt.com/docs/build/sources) and built-in [data tests](https://docs.getdbt.com/docs/build/data-tests). A working connection would require a Databricks workspace, suitable compute such as a SQL warehouse, adapter configuration, and permissions; see the official [dbt Databricks setup](https://docs.getdbt.com/docs/local/connect-data-platform/databricks-setup). These are **implementation requirements for a future integration**, not steps required to run this MVP. This proposed architecture is an inference from those platform capabilities and this project's current data flow.

## Limits and next class-project work

The source data and labels are synthetic. Only sampled primary-period rows were used for fitting and evaluation, though behavioral history is complete for those rows. The late-period shift is unexplained. Currency groups with a handful of positives have unstable AP estimates. The model has no prospective validation or probability calibration, and the bootstrap interval does not capture account dependence.

For a final class report, the most useful follow-ups are full-primary-period fitting if compute permits, repeated time windows or seeds, and case-level explanation review. If Databricks/dbt is required by the course, build and validate that engineering layer **without changing the ML evaluation boundary**, then document its feature-parity results. Graph models, live streaming, and a separate dashboard are outside this MVP.
