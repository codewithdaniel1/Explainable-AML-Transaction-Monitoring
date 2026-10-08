# Guide to the AML modeling notebook

The [notebook](../aml_modeling.ipynb) contains the code and saved outputs for this **simple ML class MVP**. This guide explains the steps and how to read the three-model comparison. Use the [README](../README.md) for installation and data download.

## Research question

Can logistic regression, XGBoost, and an Explainable Boosting Machine (EBM) rank IBM's **synthetic** laundering-labeled transactions for review? How do their ranking performance and explanations differ? A high model score suggests a transaction to inspect; it does not prove laundering.

## Follow the notebook from top to bottom

| Section | Action | Main idea |
| --- | --- | --- |
| **1–2** | Import libraries and read a reproducible 10% sample of IBM HI-Small rows before September 11. | Keep the real-data example manageable in one notebook. |
| **3** | Create amount, hour, same-bank, and earlier-sender-count features. | Add one simple behavior signal without using labels. |
| **4** | Split chronologically and encode payment format and currency. | Fit on September 1–6, choose a model on September 7–8, and report September 9–10 separately. |
| **5** | Fit the three classifiers on training rows. | Compare a linear model, a boosted tree model, and an additive interpretable model. |
| **6** | Calculate AP, ROC-AUC, Precision@100, and Recall@100; show the top-ranked test rows. | Measure ranking quality and a fixed 100-alert review queue. |
| **7** | Show logistic coefficients, one XGBoost Tree SHAP explanation, and EBM term importance. | See what the fitted models used to assign scores. |
| **8** | Read the limits. | Keep conclusions tied to sampled synthetic data. |

Tables, charts, and scores are displayed in the `.ipynb` file. The raw CSV remains at `data/raw/HI-Small_Trans.csv`; no metrics JSON, predictions CSV, or `results/` directory is created.

## Data and the time split

The source CSV has about **5.08 million** transactions and very few positive synthetic labels. The notebook samples **10%** of transactions before September 11 using seed 42, producing **507,333 rows and 475 positives** for this run. Training has **324,521 rows / 260 positives**; validation has **96,746 / 112**; the test period has **86,066 / 103**.

The time boundaries are fixed before fitting: September 1–6 for training, September 7–8 for validation, and September 9–10 for the held-out test. The original CSV's September 11–18 tail has an unusual volume and label-rate shift, so this MVP leaves it out. The notebook uses validation **average precision** to choose a model, then reads its test result.

## What the features mean

- `log_amount` is the current amount paid after a logarithm makes very large amounts less dominant.
- `hour` is the transaction's hour, and `same_bank` indicates whether sender and receiver banks match.
- `payment_format` and `payment_currency` become numeric indicator columns. Training categories define the columns; validation and test are aligned to them.
- `prior_sender_count` counts earlier **sampled** transactions from the same bank/account sender. Transactions at the same timestamp share the same prior count.

The sender count is **not complete account history**: roughly 90% of earlier source rows were not loaded. It is a classroom-scale behavior feature. The prior full-history DuckDB feature experiment is no longer part of this simplified notebook, and its old performance numbers do not describe the current model.

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
| Logistic regression | 0.0057 | 0.0055 | 0 |
| **XGBoost** | **0.0432** | **0.0334** | **5** |
| EBM | 0.0179 | 0.0225 | 4 |

Validation AP selects XGBoost. On the sampled September 9–10 test period, **5 of the top 100** rows have positive labels: 5% precision and **4.85% recall** among 103 positives. The notebook prints the full metric table. These results are lower than the previous, more complex full-history experiment and should not be mixed with its figures.

Logistic coefficients and EBM terms describe the fitted models globally. The Tree SHAP values break down **one XGBoost score** on its raw log-odds scale. Neither type of explanation establishes why a synthetic label was assigned or whether a real transaction involved laundering.

## What is outside this MVP

The model evaluates sampled synthetic rows only. The sender-history count is incomplete, the late-period shift is not evaluated, and there is no calibration or prospective validation. A class report should state these limits and treat the notebook as a comparison exercise, not an operational AML system.

### Databricks and dbt

The original proposal mentioned Databricks and dbt, but **neither runs in this notebook**. A later engineering version could load the IBM CSV into Databricks, use dbt SQL models to clean and test the table, and pass a model-ready feature table to the same three-model comparison. Databricks explains that [dbt transforms data after it is loaded](https://docs.databricks.com/aws/en/partners/prep/dbt); dbt documents how to declare [sources](https://docs.getdbt.com/docs/build/sources) and add [data tests](https://docs.getdbt.com/docs/build/data-tests).

Before using such a version for ML, compare its row counts, labels, dates, and features with the notebook. In particular, a full-data sender-history feature would **not** match this sampled-history feature, so the experiment and reported results would need to be rerun. This is a possible next step, not a claim that cloud integration is already implemented.
