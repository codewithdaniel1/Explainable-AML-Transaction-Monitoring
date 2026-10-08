# Guide to the AML modeling notebook

The [notebook](../aml_modeling.ipynb) contains the code and saved outputs for this one-notebook ML class MVP. The [README](../README.md) covers setup and the data download. This guide explains the modeling choices, the revised metrics, and what the results can support.

## Research question and notebook order

Can logistic regression, XGBoost, and an Explainable Boosting Machine (EBM) rank IBM's **synthetic** laundering-labeled transactions for review? A high score suggests a row to inspect; it does not prove laundering.

The real purpose of AML monitoring is to find potentially suspicious behavior early enough for investigators to assess it and, when appropriate, provide useful reports to authorities. A model should help cover the institution's important risks while producing alerts investigators can reasonably review. Classifier scores alone cannot establish that outcome.

| Section | What happens | Why |
| --- | --- | --- |
| **1–2** | Load a reproducible 50% sample of transactions before September 11. | Use more labeled and normal activity while keeping a laptop-friendly notebook. |
| **3** | Create current-transaction fields, earlier sender/receiver/pair counts, and the sender's earlier typical payment amount. | Capture prior behavior without using labels. |
| **4** | Split by time and encode categories. | Train on September 1–6, select on September 7–8, and report September 9–10 separately. |
| **5** | Fit the three models using training rows. | Compare linear, tree, and additive approaches on the same encoded columns. |
| **6** | Show average precision, ROC-AUC, precision, recall, F1, false positive rate, and a confusion matrix. | Evaluate ranking and binary alerts using thresholds selected on validation data. |
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
- **Earlier sampled activity:** `prior_sender_count`, `prior_receiver_count`, and `prior_pair_count` count earlier transactions for the sender, receiver, and sender–receiver relationship. `log_prior_sender_mean_amount` is the log of one plus that sender's average earlier payment in the same currency; it is zero with no such history. Rows with the same timestamp do not count one another.
- **Account and bank IDs:** kept in the working `transactions` table and used to form history and relationship features. The models do not receive raw IDs as predictor columns.

All **11 original CSV fields** are read. Here is where each one goes:

| Source field | Use in this notebook |
| --- | --- |
| `Timestamp` | Time split, transaction hour, and strictly earlier history. |
| `From Bank`, sender `Account` | Sender identity for past-activity counts and same-bank/account flags. |
| `To Bank`, receiver `Account` (`Account.1` in pandas) | Receiver and sender–receiver history; same-bank/account flags. |
| `Amount Paid`, `Payment Currency` | Log paid amount, payment-currency category, and earlier same-currency average amount for the sender. |
| `Amount Received`, `Receiving Currency` | Log received amount and receiving-currency category. |
| `Payment Format` | Encoded payment-format category. |
| `Is Laundering` | Training/evaluation label only; never a predictor. |

These history features are incomplete because half of eligible transactions were left out. They are a simple approximation of account behavior and network context. The [IBM paper](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) uses richer graph features, including vertex statistics and cycle patterns. It also **excludes raw account IDs as model predictors** to prevent learning the IDs themselves. That is a modeling choice, not a regulatory ban. Its published F1 results use a different feature pipeline and split, so they are **not directly comparable** with this notebook's F1.

## Why the old results were weak

Adding rows alone did little. With the **old features and old inverse-frequency XGBoost weight**, changing the sample from 25% to 50% moved test average precision from **0.0366 to 0.0367**. The old weight at 50% was about **1,289**. In exploratory validation checks, a weight of **10** ranked better, so the notebook now uses 10 for all three models.

At 50% sampling and XGBoost weight 10, these controlled feature checks show where the gain came from. They used the same dates and model settings; the reporting period was examined during development.

| Features | Validation AP | Reporting-period AP |
| --- | ---: | ---: |
| Previous simple features | 0.0602 | 0.0535 |
| Plus received amount, currency, and same-account flag | 0.0627 | 0.0576 |
| Plus receiver and pair history counts | 0.2164 | 0.1441 |
| All previous transaction and history features | 0.2349 | 0.1525 |
| Previous features plus sender's earlier same-currency mean payment | **0.4427** | **0.3282** |

Earlier relationship activity and the sender's **typical earlier payment size** provided the largest gains. On validation, the one added average-amount feature raised XGBoost F1 from **0.2965** to **0.4766**. We also tried daily activity counts and distinct-counterparty counts; they did not improve this validation setup, so this MVP keeps one new feature. These comparisons do not prove the same improvement on another dataset or a truly fresh time period.

## What performance should this project aim for?

**There is no universal regulatory minimum** such as “AML models must have 80% precision” or a fixed recall, F1, or false positive rate. The [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04) describes monitoring that fits a bank's risk profile, followed by investigation and periodic review of thresholds. The [Wolfsberg Group's 2025 monitoring statement](https://wolfsberg-group.org/resources/195/202) lists precision and recall **for consideration alongside** priority-risk coverage, broader risk indicators, SAR quality feedback, and downstream investigation. The [2026 U.S. interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) calls for testing aligned with model purpose and for validation and monitoring; it does not set AML classifier percentages. Those sources support bank-specific acceptance criteria, not a single industry cutoff.

For **this synthetic class project**, we set this illustrative goal *before running the revised feature on September 9–10*:

| Measure on September 9–10 | Class-project goal | Why it matters |
| --- | ---: | --- |
| Precision | At least **30%** | At least 3 of 10 alerts match a positive synthetic label. |
| Recall | At least **30%** | Detect at least 3 of 10 positive synthetic labels. |
| F1 | At least **0.30** | Keep precision and recall reasonably balanced. |
| False positive rate | Below **0.1%** | Fewer than 10 false alerts per 10,000 normal transactions. |

These four percentages are **our educational targets**, not a bank's required thresholds. Even a 0.1% false positive rate can create many false alerts at bank scale, so the notebook also shows **alert count** and the confusion matrix. In actual AML work, labels are incomplete and SAR quality, investigator capacity, risk coverage, and missed-risk reviews would matter as well. Here, `Is Laundering` is a complete synthetic label for this exercise; it is not a confirmed SAR outcome.

## Read the revised results

The notebook uses established classification measures: **average precision (AP)** and **ROC-AUC** for ranking, and **precision, recall, F1, false positive rate, and a confusion matrix** for alerts at an explicit threshold. Precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, F1 is their harmonic mean, and false positive rate is `FP / (FP + TN)`. The [IBM benchmark](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) emphasizes minority-class F1, precision, and recall because accuracy is misleading when labels are rare. The reporting-period positive-label rate is **0.00113**, the random-ranking baseline for AP. [scikit-learn's precision–recall guide](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) explains AP and the precision–recall tradeoff.

For each model, the notebook chooses the threshold with the best **validation F1**, then applies that same threshold to the later reporting period. It chooses the model with the best validation AP. The later-period results are:

| Model | Validation AP | Later AP | Later ROC-AUC | Later precision | Later recall | Later F1 | Later FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.0303 | 0.0171 | 0.9423 | 0.0232 | 0.4529 | 0.0441 | 2.165% |
| **XGBoost** | **0.4427** | **0.3282** | **0.9772** | 0.3576 | **0.3627** | **0.3601** | 0.0738% |
| EBM | 0.3829 | 0.2756 | 0.9711 | **0.4753** | 0.2561 | 0.3329 | **0.0320%** |

Validation AP selects XGBoost. Its validation-selected threshold is **0.3109**. On September 9–10 it raises **495 alerts**: **177 true positives**, **318 false positives**, and **311 missed positive labels**. Its false positive rate is `318 / 430,618 = 0.0738%`. All four classroom goals are met on this later period. Recall is still only **36.3%** of known synthetic positives, so meeting the goal does not imply complete coverage.

All three models use the same **46 encoded features** and positive-class weight of 10. Logistic regression is linear; XGBoost can learn interactions; EBM uses nonlinear additive effects with interactions disabled. We show logistic coefficients, one XGBoost Tree SHAP breakdown, and EBM term importance in section 7. Explanations describe model behavior, not why IBM assigned a label. A bank would set its alert threshold using its own risk and review capacity; maximizing F1 is a transparent classroom choice, **not a regulatory requirement**.

## Limits and bank context

This is an exploratory comparison on sampled synthetic data. The history counts omit half of eligible rows, September 11–18 is excluded, the reporting period was consulted during revisions, and model scores are **not calibrated probabilities**. No score or synthetic label is an investigator decision. These are meaningful limitations for a class report.

The CSV has synthetic laundering labels, but no investigation outcomes or SAR decisions. The notebook can calculate classifier precision and recall against those synthetic labels; it cannot measure a bank's actual SAR conversion rate or investigation quality. The [IBM dataset paper](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) notes that its complete synthetic ground truth differs from real AML data, where many laundering transactions are never detected.

Banks tailor suspicious-activity monitoring to their risks. They may combine transaction reports, rules or intelligent surveillance, and referrals; investigators then review customer and transaction context before making a Suspicious Activity Report decision. The [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04) describes this broader workflow. The notebook covers only transaction ranking. Current [interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) also discusses appropriate testing, limitations, and ongoing monitoring for bank models.

Adding relevant behavior features is normal in AML monitoring: the FFIEC manual describes systems that compare activity with account history and peers. It does **not** prescribe using every raw column or a fixed number of features. It expects monitoring criteria to match the bank's risk profile and to be reviewed and tested. The [2026 interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) calls for data and method choices aligned with model purpose, attention to input quality, and appropriate validation. The CSV has no customer-risk, occupation, geography, or due-diligence data; those cannot be honestly added from this source.

### Databricks and dbt

The original proposal mentioned Databricks and dbt, but **neither runs in this one-notebook MVP**. A later engineering version could load the IBM CSV into Databricks and use dbt to clean and check a model-ready table. See [Databricks' dbt integration](https://docs.databricks.com/aws/en/partners/prep/dbt) and [dbt's source and test documentation](https://docs.getdbt.com/docs/build/data-tests). Any such version would need to reproduce the notebook's time boundaries and strictly earlier history features before its results could be compared.
