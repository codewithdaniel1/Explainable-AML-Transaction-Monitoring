# Guide to the AML modeling notebook

The [notebook](../aml_modeling.ipynb) contains the code and saved outputs for this **simple ML class MVP**. This guide explains the steps and how to read the three-model comparison. Use the [README](../README.md) for installation and data download.

## Research question

Can logistic regression, XGBoost, and an Explainable Boosting Machine (EBM) rank IBM's **synthetic** laundering-labeled transactions for review? How do their ranking performance and explanations differ? A high model score suggests a transaction to inspect; it does not prove laundering.

## Follow the notebook from top to bottom

| Section | Action | Main idea |
| --- | --- | --- |
| **1–2** | Import libraries and read a reproducible 25% sample of IBM HI-Small rows before September 11. | Use more data while keeping the example manageable in one notebook. |
| **3** | Create amount, hour, same-bank, and earlier-sender-count features. | Add one simple behavior signal without using labels. |
| **4** | Split chronologically and encode payment format and currency. | Fit on September 1–6, choose a model on September 7–8, and report September 9–10 separately. |
| **5** | Fit the three classifiers on training rows. | Compare a linear model, a boosted tree model, and an additive interpretable model. |
| **6** | Calculate AP, ROC-AUC, Precision@100, and Recall@100; show the top-ranked test rows. | Measure ranking quality and a fixed 100-alert review queue. |
| **7** | Show logistic coefficients, one XGBoost Tree SHAP explanation, and EBM term importance. | See what the fitted models used to assign scores. |
| **8** | Read the limits. | Keep conclusions tied to sampled synthetic data. |

Tables, charts, and scores are displayed in the `.ipynb` file. The raw CSV remains at `data/raw/HI-Small_Trans.csv`; no metrics JSON, predictions CSV, or `results/` directory is created.

## Data and the time split

The source CSV has about **5.08 million** transactions and very few positive synthetic labels. The notebook samples **25%** of transactions before September 11 using seed 42, producing **1,269,729 rows and 1,162 positives** for this run. Training has **812,566 rows / 654 positives**; validation has **241,813 / 271**; the test period has **215,350 / 237**.

The time boundaries are fixed before fitting: September 1–6 for training, September 7–8 for validation, and September 9–10 for the held-out test. The original CSV's September 11–18 tail has an unusual volume and label-rate shift, so this MVP leaves it out. The notebook uses validation **average precision** to choose a model, then reads its test result.

This is a **time-based evaluation**, not a time-series forecasting model. A bank would apply a fitted monitoring system to activity arriving after training. Keeping validation and test dates later than training better represents that use. A random row split can put later behavior in training while testing on earlier, related transactions, making the result less credible. The `prior_sender_count` feature uses only earlier sampled transactions, including earlier history available when a validation or test row occurs. Current [Federal Reserve model-risk guidance](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm) lists out-of-time testing among model testing approaches; the exact split and controls depend on the model's purpose and risk.

## What the features mean

- `log_amount` is the current amount paid after a logarithm makes very large amounts less dominant.
- `hour` is the transaction's hour, and `same_bank` indicates whether sender and receiver banks match.
- `payment_format` and `payment_currency` become numeric indicator columns. Training categories define the columns; validation and test are aligned to them.
- `prior_sender_count` counts earlier **sampled** transactions from the same bank/account sender. Transactions at the same timestamp share the same prior count.

The sender count is **not complete account history**: roughly 75% of earlier eligible rows were not loaded. It is a classroom-scale behavior feature.

## Why these three models

| Model | Simple description | Explanation shown |
| --- | --- | --- |
| **Logistic regression** | A weighted linear classifier; its input columns are scaled before fitting. | Coefficients for the largest fitted effects. |
| **XGBoost** | Weighted boosted decision trees that can learn nonlinear combinations of inputs. | Native Tree SHAP contributions for one high-scored test row. |
| **EBM** | Weighted additive, nonlinear feature effects, with interactions disabled here. | Global importance of the learned terms. |

All three see the **same 26 encoded feature columns** and the same calendar windows. Positive training labels are rare, so the models give them more weight. That helps fitting but means a score such as `0.8` should **not** be read as an 80% real-world laundering probability.

## Read the results

**Average precision (AP)** summarizes precision across recall levels and is the selection measure. **ROC-AUC** is another ranking measure. **Precision@100** asks what share of the top 100 transactions have positive labels. **Recall@100** asks what share of all positives in that period appear in those 100.

| Model | Validation AP | Test AP | Test positives in top 100 |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.0057 | 0.0052 | 0 |
| **XGBoost** | **0.0406** | **0.0366** | **7** |
| EBM | 0.0267 | 0.0244 | 1 |

Validation AP selects XGBoost. On the sampled September 9–10 test period, **7 of the top 100** rows have positive labels: 7% precision and **2.95% recall** among 237 positives. The notebook prints the full metric table. Increasing the sample changes both the training data and the sampled-history feature, so these figures should not be compared as though only sample size changed.

Logistic coefficients and EBM terms describe the fitted models globally. The Tree SHAP values break down **one XGBoost score** on its raw log-odds scale. Neither type of explanation establishes why a synthetic label was assigned or whether a real transaction involved laundering.

## How bank AML monitoring relates to this exercise

Banks tailor suspicious-activity monitoring to their risks. A common workflow is to combine transaction reports, rules or intelligent surveillance, and referrals; generate alerts; have investigators examine transaction and customer context; and decide whether activity warrants a Suspicious Activity Report. Monitoring may use rolling periods, customer history, and peer comparisons. Thresholds and systems are reviewed and tested over time. The [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04) describes these approaches and notes that processes vary by bank size and risk profile.

For this class project, the model score is an **alert ranking** for a fixed review capacity of 100 transactions. It is not a replacement for investigation, a SAR decision, or a production validation. A bank's model process would also examine data quality, later-period performance, limitations, and ongoing monitoring under the [2026 interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm). This bank workflow is context for the exercise, not a claim that all banks use these three models.

## What is outside this MVP

The model evaluates sampled synthetic rows only. The sender-history count is incomplete, the late-period shift is not evaluated, and there is no calibration or prospective validation. A real monitoring team would also assess the excluded shift, rather than assume earlier-period results still hold. A class report should state these limits and treat the notebook as a comparison exercise, not an operational AML system.

### Databricks and dbt

The original proposal mentioned Databricks and dbt, but **neither runs in this notebook**. A later engineering version could load the IBM CSV into Databricks, use dbt SQL models to clean and test the table, and pass a model-ready feature table to the same three-model comparison. Databricks explains that [dbt transforms data after it is loaded](https://docs.databricks.com/aws/en/partners/prep/dbt); dbt documents how to declare [sources](https://docs.getdbt.com/docs/build/sources) and add [data tests](https://docs.getdbt.com/docs/build/data-tests).

Before using such a version for ML, compare its row counts, labels, dates, and features with the notebook. In particular, a full-data sender-history feature would **not** match this sampled-history feature, so the experiment and reported results would need to be rerun. This is a possible next step, not a claim that cloud integration is already implemented.
