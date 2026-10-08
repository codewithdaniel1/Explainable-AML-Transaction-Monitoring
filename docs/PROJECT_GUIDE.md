# Guide to the AML modeling notebook

[aml_modeling.ipynb](../aml_modeling.ipynb) is the project: it contains the code, tables, charts, and a synthetic case review. This guide explains why each step exists and how to read the model comparison. Use the [README](../README.md) for setup and data download.

## The question

Can a model rank synthetic laundering-labeled transactions for a fixed review queue, and how do performance and explanations differ between logistic regression, XGBoost, and an Explainable Boosting Machine (EBM)? Currency-aware rules provide a simple baseline. A score recommends **review**, not a finding that laundering occurred.

## Read the notebook in order

| Notebook sections | What happens | Why it matters |
| --- | --- | --- |
| **0–1: Setup and profile** | Check the local IBM CSV, scan all 5,078,345 rows, plot daily volume and label rate, and sample 10% of the primary period. | Establishes class imbalance and reveals the unusual final eight days. |
| **2: Time windows** | Train on September 1–6, validate on September 7–8, and hold out September 9–10. Keep September 11–18 separate. | Later labels do not enter model fitting or model selection. |
| **3: Earlier-history features** | DuckDB computes 24-hour and 7-day activity from **all original earlier transactions** for each sampled row. | Preserves account history even though modeling rows are sampled. |
| **4–5: Models and fitting** | Fit rules plus three ML algorithms using transaction-only and behavioral inputs. Select the specification with the highest validation average precision. | Makes seven comparisons under the same calendar design. |
| **6–7: Performance and errors** | Show comparison charts, precision–recall curves, top alerts, missed positives, currency groups, and a descriptive bootstrap interval. | Connects ranking quality to a fixed 100-alert review capacity. |
| **8: Explanations** | Show logistic coefficients, XGBoost importance and Tree SHAP, and EBM terms. | Shows what each fitted model used when scoring. |
| **9–10: Replay and case review** | Replay saved scores in memory and inspect one ranked synthetic transaction with its earlier sender activity. | Demonstrates the review workflow in the notebook without a separate app or output files. |
| **11: Limits** | State what the sampled experiment supports and what remains open. | Prevents treating a class pilot as a deployed AML system. |

The raw CSV stays at `data/raw/HI-Small_Trans.csv`. The notebook displays results inside the `.ipynb` file; it does not write a `results/` folder.

## Data and features

The target is IBM's synthetic `is_laundering` label. The full CSV has 5,177 positive labels among 5,078,345 transactions (about 0.102%). The September 11–18 tail contains only 1,108 rows, including 655 positives, so the notebook treats it as a **stress check**. Its label rate is far higher than the primary period's; average precision from those periods should not be compared directly.

**Transaction features** describe the current payment: log amount paid, hour, day of week, same-bank indicator, payment format, and payment and receiving currencies. **Behavioral features** add the sender's earlier transaction counts over 24 hours and seven days, recent paid amount within the same payment currency, amount relative to that currency's earlier average, and whether the receiving account is new within seven days. Bank and account IDs build history but are not direct model inputs.

All history uses strictly earlier timestamps. Transactions at the same timestamp cannot see one another. Amounts are not summed across different currencies. The 10% sample is used for fitting and evaluation; DuckDB still reads all earlier source transactions to build history.

## What each model contributes

| Specification | How it works | How to read its explanation |
| --- | --- | --- |
| **Currency-aware rules** | Combines a training-only 99th-percentile amount cutoff for each payment currency with prior-activity thresholds. No ML model is fitted. | Each rule is explicit; the thresholds are illustrative, not validated bank policy. |
| **Logistic regression** | Fits a weighted linear classifier after scaling numeric inputs and encoding categories. | Coefficients show direction and size on the fitted log-odds scale, subject to the preprocessing. |
| **XGBoost** | Fits weighted boosted decision trees that can capture feature interactions. | Feature importance summarizes the fitted trees; Tree SHAP breaks down selected transaction scores into log-odds contributions. |
| **EBM** | Fits weighted additive, nonlinear feature effects, with interactions disabled in this pilot. | Global terms show which inputs contribute most across the fitted model. |

Each ML algorithm runs once with transaction-only inputs and once with behavioral inputs. Together with the rules, that gives **seven specifications**. Explanations describe model calculations; they do not establish why a transaction was labeled or prove real-world laundering.

## How to interpret the comparison

The notebook uses **average precision (AP)** as its main selection metric because positive labels are rare. The code key `pr_auc` stores scikit-learn's `average_precision_score`. **Precision@100** is the share of positive labels in the 100 highest-scored transactions; **Recall@100** is the share of all positives found within those 100. ROC-AUC and Brier score are also computed, but weighted scores are not calibrated probabilities.

Validation AP selects **XGBoost with behavioral features**. On the sampled September 9–10 holdout, it has AP **0.2245** and finds **23 of 103** positive labels in the top 100 alerts (23% precision; 22.3% recall). A 300-repeat stratified row bootstrap gives a descriptive AP interval of **0.1468–0.3128**. That interval does not account for transactions sharing accounts or for time dependence. The notebook contains the full seven-specification table and figures.

## Limits of this MVP

The IBM data and labels are synthetic. The model evaluation uses sampled primary-period rows, even though the history features use the full earlier CSV. The cause of the late-period volume and label shift is unknown. Small currency groups have unstable metrics, and no score calibration or prospective operational validation has been done.

For the class report, the next useful steps are to test full-primary-period fitting if compute permits, repeat the comparison across seeds or time windows, inspect example explanations, and state the sampling and synthetic-data limits. Databricks/dbt, graph models, live streaming, and a separate dashboard are outside this notebook MVP.
