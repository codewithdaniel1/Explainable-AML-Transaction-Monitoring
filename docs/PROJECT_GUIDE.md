# Guide to the AML modeling notebook

The [notebook](../aml_modeling.ipynb) contains the code and saved outputs for this one-notebook ML class MVP. The [README](../README.md) covers setup and the data download. This guide explains the modeling choices, the revised metrics, and what the results can support.

## Research question and notebook order

Can logistic regression, XGBoost, and an Explainable Boosting Machine (EBM) rank IBM's **synthetic** laundering-labeled transactions for review? A high score suggests a row to inspect; it does not prove laundering.

| Section | What happens | Why |
| --- | --- | --- |
| **1–2** | Load a reproducible 50% sample of transactions before September 11. | Use more labeled and normal activity while keeping a laptop-friendly notebook. |
| **3** | Create current-transaction fields and earlier sender, receiver, and sender–receiver counts. | Capture a small part of the transaction network without using labels. |
| **4** | Split by time and encode categories. | Train on September 1–6, select on September 7–8, and report September 9–10 separately. |
| **5** | Fit the three models using training rows. | Compare linear, tree, and additive approaches on the same 45 encoded columns. |
| **6** | Show average precision, ROC-AUC, and precision and recall at 100, 500, and 1,000 alerts. | Compare overall ranking and several review capacities. |
| **7–8** | Show model explanations and limits. | Explain scores without treating them as conclusions about a real customer. |

The raw CSV remains at `data/raw/HI-Small_Trans.csv`. Tables and charts stay in the notebook; it creates no metrics JSON, predictions CSV, or `results/` directory.

## Data and time split

The HI-Small CSV contains **5,078,345** transactions. This run samples 50% of rows before September 11 using seed 42: **2,539,464 rows and 2,271 positive labels**. The fixed windows are:

| Period | Dates in 2022 | Rows | Positive labels |
| --- | --- | ---: | ---: |
| Training | September 1–6 | 1,625,750 | 1,260 |
| Validation | September 7–8 | 482,608 | 523 |
| Reporting period | September 9–10 | 431,106 | 488 |

This is a **time-based evaluation**, not a time-series forecasting model. Training on earlier transactions and evaluating on later activity resembles the intended use of monitoring. The original [IBM benchmark paper](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) also uses a temporal split and computes graph features from past activity to avoid future information reaching earlier rows.

We exclude September 11–18 because those eight days contain only **1,108 rows, 655 labeled positive**, a sharp change from the preceding millions of rows. This means the notebook does **not** show how a model handles that shift. A real monitoring process would need to examine it. The reporting period was also inspected during this project's feature and weighting revisions, so the current figures are **exploratory**, not an untouched final estimate.

## What the model sees

- **Current transaction:** log paid amount, log received amount, hour, same-bank flag, same-account flag, payment format, payment currency, and receiving currency.
- **Earlier sampled activity:** `prior_sender_count`, `prior_receiver_count`, and `prior_pair_count` count earlier transactions for the sender, receiver, and sender–receiver relationship. Rows with the same timestamp do not count one another.
- **Account IDs:** used to form the history counts, then dropped. The models do not receive raw account IDs as columns.

These history counts are incomplete because half of eligible transactions were left out. They are a simple approximation of network context. The [IBM paper](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) uses richer graph features, including vertex statistics and cycle patterns. Its published F1 results use a different feature pipeline and metric, so they are **not directly comparable** with this notebook's average precision.

## Why the old results were weak

Adding rows alone did little. With the **old features and old inverse-frequency XGBoost weight**, changing the sample from 25% to 50% moved test average precision from **0.0366 to 0.0367**. The old weight at 50% was about **1,289**. In exploratory validation checks, a weight of **10** ranked better, so the notebook now uses 10 for all three models.

At 50% sampling and XGBoost weight 10, these controlled feature checks show where the gain came from. They used the same dates and model settings; the reporting period was examined during development.

| Features | Validation AP | Reporting-period AP | Positives in top 100 |
| --- | ---: | ---: | ---: |
| Previous simple features | 0.0602 | 0.0535 | 10 |
| Plus received amount, currency, and same-account flag | 0.0627 | 0.0576 | 9 |
| Plus receiver and pair history counts | 0.2164 | 0.1441 | 44 |
| All revised features | **0.2349** | **0.1525** | **45** |

The main missing signal was **earlier relationship activity**, not just more rows or extra current-transaction fields. This table describes these experiments; it does not prove the same improvement on another dataset or a truly fresh time period.

## Read the revised results

**Average precision (AP)** summarizes precision across recall levels. The reporting-period positive-label rate is **0.00113**, so a random ranking has AP around **0.00113**. ROC-AUC is a secondary ranking measure and can look high even when the top alert queue is weak. [scikit-learn's precision–recall guide](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) explains the metrics.

| Model | Validation AP | Reporting-period AP | Reporting-period ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.0302 | 0.0172 | 0.9417 |
| **XGBoost** | **0.2349** | **0.1525** | **0.9759** |
| EBM | 0.1294 | 0.0556 | 0.9674 |

Validation AP selects XGBoost. All three models use the same 45 encoded features and positive-class weight of 10. Logistic regression is linear; XGBoost can learn interactions; EBM uses nonlinear additive effects with interactions disabled. We show logistic coefficients, one XGBoost Tree SHAP breakdown, and EBM term importance in section 7. Explanations describe model behavior, not why IBM assigned a label.

**Top 100 is one example review capacity, not the best universal AML metric.** Here each validation or reporting window spans two days. Precision is the share of reviewed alerts with positive labels; recall is the share of all positive labels found. The notebook displays the tradeoff for XGBoost:

| Reporting-period alerts reviewed | Positives found | Precision | Recall |
| ---: | ---: | ---: | ---: |
| 100 | 45 | 45.0% | 9.22% |
| 500 | 104 | 20.8% | 21.31% |
| 1,000 | 152 | 15.2% | 31.15% |

A team would choose its alert budget from actual analyst capacity and review cost, often for a defined day or week. For the class comparison, use **AP** as the main model-ranking metric and show **precision and recall at several plausible budgets**. Report the positive-label rate as a baseline. Avoid accuracy as the headline metric for these rare labels.

## Limits and bank context

This is an exploratory comparison on sampled synthetic data. The history counts omit half of eligible rows, September 11–18 is excluded, the reporting period was consulted during revisions, and model scores are **not calibrated probabilities**. No score or synthetic label is an investigator decision. These are meaningful limitations for a class report.

Banks tailor suspicious-activity monitoring to their risks. They may combine transaction reports, rules or intelligent surveillance, and referrals; investigators then review customer and transaction context before making a Suspicious Activity Report decision. The [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04) describes this broader workflow. The notebook covers only transaction ranking. Current [interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) also discusses appropriate testing, limitations, and ongoing monitoring for bank models.

### Databricks and dbt

The original proposal mentioned Databricks and dbt, but **neither runs in this one-notebook MVP**. A later engineering version could load the IBM CSV into Databricks and use dbt to clean and check a model-ready table. See [Databricks' dbt integration](https://docs.databricks.com/aws/en/partners/prep/dbt) and [dbt's source and test documentation](https://docs.getdbt.com/docs/build/data-tests). Any such version would need to reproduce the notebook's time boundaries and strictly earlier history features before its results could be compared.
