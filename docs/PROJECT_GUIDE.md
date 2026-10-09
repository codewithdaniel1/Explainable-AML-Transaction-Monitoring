# Guide to the AML modeling notebook

The [notebook](../aml_modeling.ipynb) contains the code and saved outputs for this one-notebook ML class MVP. The [README](../README.md) covers setup and the data download. This guide explains the modeling choices, the revised metrics, and what the results can support.

## Research question and notebook order

Can logistic regression, XGBoost, and an Explainable Boosting Machine (EBM) rank IBM's **synthetic** laundering-labeled transactions for review? A high score suggests a row to inspect; it does not prove laundering.

The real purpose of AML monitoring is to find potentially suspicious behavior early enough for investigators to assess it and, when appropriate, provide useful reports to authorities. A model should help cover the institution's important risks while producing alerts investigators can reasonably review. Classifier scores alone cannot establish that outcome.

| Section | What happens | Why |
| --- | --- | --- |
| **1–2** | Load a reproducible 50% modeling sample and retain all eligible earlier transactions for history. | Keep model fitting manageable while preserving account activity. |
| **3** | Create current-transaction fields, earlier sender/receiver/pair counts, the sender's earlier typical payment amount, and candidate short-window features. | Capture past behavior without using labels or future transactions. |
| **4** | Split by time and encode categories. | Train on September 1–6, select on September 7–8, and report September 9–10 separately; show row-count percentages. |
| **5** | Define logistic regression, XGBoost, and EBM. | Hold their settings constant across time folds. |
| **6** | Run three expanding time folds inside September 1–6. | Compare models on later transactions without mixing future and past rows. |
| **7** | Fit the three models on all September 1–6 training rows. | Use all available training data after the fold comparison. |
| **8** | Select alert thresholds on September 7–8, report September 9–10, compare short-window candidates, and inspect misses by payment format and day. | Make feature decisions and coverage gaps visible. |
| **9** | Give each model its own result table, four-metric chart, and feature explanation. | Make precision–recall tradeoffs and alert workload visible. |
| **10** | Show the experiment's limits. | Keep synthetic scores separate from real AML decisions. |

The raw CSV remains at `data/raw/HI-Small_Trans.csv`. Tables and charts stay in the notebook; it creates no metrics JSON, predictions CSV, or `results/` directory.

## Data and time split

The HI-Small CSV contains **5,078,345** transactions. This run samples 50% of rows before September 11 using seed 42: **2,539,464 rows and 2,271 positive labels**. The fixed windows are:

| Period | Dates in 2022 | Rows | Share of modeled sample | Positive labels | Positive rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Training | September 1–6 | 1,625,750 | **64.02%** | 1,260 | **0.0775%** |
| Validation | September 7–8 | 482,608 | **19.00%** | 523 | **0.1084%** |
| Later test/reporting | September 9–10 | 431,106 | **16.98%** | 488 | **0.1132%** |

These shares are computed from the **2,539,464 sampled eligible rows**, not from the entire raw CSV. The fixed date boundaries produce about 64/19/17 rather than an exact 60/20/20 split. This is a **time-based evaluation**, not a time-series forecasting model. Training on earlier transactions and evaluating on later activity resembles the intended use of monitoring. The original [IBM benchmark paper](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) also uses a temporal split and computes graph features from past activity to avoid future information reaching earlier rows.

### Label balance

`Is Laundering` is strongly imbalanced. The notebook shows both sides of the label split:

| Population | Rows | Positive labels | Positive rate | Negative labels | Negative rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full raw CSV | 5,078,345 | 5,177 | **0.1019%** | 5,073,168 | 99.8981% |
| Eligible dates before September 11 | 5,077,237 | 4,522 | **0.0891%** | 5,072,715 | 99.9109% |
| Modeled 50% sample | 2,539,464 | 2,271 | **0.0894%** | 2,537,193 | 99.9106% |

Only about **9 in 10,000** sampled transactions have a positive synthetic label. The raw CSV's positive rate is slightly higher because the excluded September 11–18 tail contains **655 positives in only 1,108 rows**. The training, validation, and test rows above have different positive rates, so average precision and alert precision need to be interpreted against each period's own prevalence. Accuracy alone would be misleading: predicting every modeled row negative would be about **99.91% accurate** while finding zero positives.

### Expanding validation folds

Within the **September 1–6 training portion**, section 6 now compares all three models on three time-ordered folds:

| Fold | Model fitting dates | Fold validation date |
| --- | --- | --- |
| 1 | September 1–3 | September 4 |
| 2 | September 1–4 | September 5 |
| 3 | September 1–5 | September 6 |

The model and category encoding are fitted from each fold's earlier rows; history features for a transaction use only activity before that transaction. The notebook compares **average precision (AP)** by fold and uses the mean to select a model. AP needs no alert threshold. Final versions of the models are then trained on **September 1–6**, alert thresholds are chosen on **September 7–8**, and **September 9–10** is reported later. This is expanding-window validation, not random or stratified K-fold. The [2026 U.S. interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) discusses out-of-time testing as one approach; it does not mandate a particular cross-validation method.

The extreme class imbalance is a reason to **check each fold's positive count**, not to mix future transactions into training. The September 4, 5, and 6 validation folds contain **196, 235, and 258 positive labels**, respectively. Their positive rates are **0.1893%, 0.0972%, and 0.1068%**. [Stratified K-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) would make fold label rates similar, but could train on later behavior and validate on earlier behavior; [time-ordered splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) avoids that reversal. We retain the changing prevalence, use class weighting when fitting, and compare ranking with AP. If a future time fold has too few positives for a meaningful estimate, we should use a longer later-time window rather than shuffle dates.

| Model | Sep 4 fold AP | Sep 5 fold AP | Sep 6 fold AP | Mean AP |
| --- | ---: | ---: | ---: | ---: |
| **XGBoost** | 0.4253 | 0.4457 | 0.4562 | **0.4424** |
| EBM | 0.3833 | 0.4114 | 0.3908 | 0.3952 |
| Logistic regression | 0.0494 | 0.0185 | 0.0313 | 0.0331 |

XGBoost has the highest mean fold AP. These fold scores compare ranking on earlier days; they are separate from the September 7–8 threshold selection and September 9–10 results.

We exclude September 11–18 because those eight days contain only **1,108 rows, 655 labeled positive**, a sharp change from the preceding millions of rows. This means the notebook does **not** show how a model handles that shift. A real monitoring process would need to examine it. The reporting period was also inspected during this project's feature and weighting revisions, so the current figures are **exploratory**, not an untouched final estimate.

## What the model sees

- **Current transaction:** log paid amount, log received amount, hour, same-bank flag, same-account flag, payment format, payment currency, and receiving currency.
- **Earlier eligible activity:** `prior_sender_count`, `prior_receiver_count`, and `prior_pair_count` count earlier transactions for the sender, receiver, and sender–receiver relationship across the full eligible history. `log_prior_sender_mean_amount` is the log of one plus that sender's average earlier payment in the same currency; it is zero with no such history. Rows with the same timestamp do not count one another.
- **Short-window candidates:** Earlier outgoing/incoming counts and amounts within fixed six-hour blocks and calendar days, plus earlier incoming amount for the sender within its current six-hour block. These are shown in a validation comparison and are included in the final models only if they help. A fixed block is not a rolling window.
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

History features use all **5,077,237** eligible transactions, including rows not selected for model fitting. They still capture only simple account and pair behavior. The [IBM paper](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) uses richer graph features, including vertex statistics and cycle patterns. It also **excludes raw account IDs as model predictors** to prevent learning the IDs themselves. That is a modeling choice, not a regulatory ban. Its published F1 results use a different feature pipeline and split, so they are **not directly comparable** with this notebook's F1.

## Why the results changed

Early versions relied on current-transaction fields and partial sampled account histories. More sampled rows alone barely helped: under the older feature and weight settings, increasing the sample from 25% to 50% moved later-period XGBoost AP from **0.0366 to 0.0367**. Adding sender, receiver, and pair behavior plus an earlier same-currency typical payment improved the model. The current run keeps those model inputs but computes their history from **all eligible earlier transactions**, rather than only sampled rows. Positive-class weight remains 10.

We also tried a modestly deeper XGBoost model. It raised validation AP from **0.4781 to 0.5196** and later-period AP from **0.4472 to 0.4640** using the same full-history inputs. The notebook compares that model with seven additional fixed six-hour and calendar-day candidate features. Both versions use the same validation rule: choose the highest recall with at least 70% precision.

| XGBoost feature set | Validation AP | Validation recall | Validation F1 | Later AP | Later recall | Later F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full earlier history | **0.5196** | **0.4417** | **0.5416** | **0.4640** | **0.4119** | **0.4933** |
| Full history plus short-window candidates | 0.5158 | 0.4149 | 0.5210 | 0.4625 | 0.4016 | 0.4612 |

The short-window candidates lowered validation AP, recall, and F1, so the final three models use the full-history inputs without them. The later period agrees with that choice, but it was examined during development and is **exploratory**, not a pristine confirmation. A model may need richer temporal or graph patterns to catch different laundering typologies; simply adding these seven columns did not do that here.

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

The notebook uses **average precision (AP)** for ranking and **precision, recall, F1, false positive rate, alert counts, and a confusion matrix** for alerts at an explicit threshold. Precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, F1 is their harmonic mean, and false positive rate is `FP / (FP + TN)`. The [IBM benchmark](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) emphasizes minority-class F1, precision, and recall because accuracy is misleading when labels are rare. The reporting-period positive-label rate is **0.00113**, the random-ranking baseline for AP. [scikit-learn's precision–recall guide](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) explains AP and the precision–recall tradeoff.

Section 9 of the notebook gives **each model its own chart** for validation and later-test precision, recall, F1, and FPR. FPR uses a separate vertical scale because it is much smaller than the other rates. AP and alert/true/false counts stay in the adjacent model-specific table. These are measures we can compute from the synthetic labels; the CSV cannot supply priority-risk coverage or SAR quality feedback recommended for real monitoring by [Wolfsberg](https://wolfsberg-group.org/resources/195/202).

The notebook chooses the model by **mean AP across the three earlier time folds**. Logistic regression and EBM use their best September 7–8 F1 thresholds. To improve detection, XGBoost uses the threshold with the highest **validation recall while retaining at least 70% validation precision**. All thresholds are applied unchanged to the later reporting period. [scikit-learn's threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html) explains why changing an alert threshold changes precision and recall without changing the model's ranking. The later-period results are:

| Model | Validation AP | Later AP | Later precision | Later recall | Later F1 | Later FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.0350 | 0.0196 | 0.0275 | **0.5164** | 0.0523 | 2.067% |
| **XGBoost** | **0.5196** | **0.4640** | 0.6147 | 0.4119 | **0.4933** | 0.0293% |
| EBM | 0.4375 | 0.3976 | **0.8114** | 0.2910 | 0.4284 | **0.0077%** |

Time-fold mean AP selects XGBoost. Its **validation precision-constrained threshold** is **0.4710**. On September 9–10 it raises **327 alerts**: **201 true positives**, **126 false positives**, and **287 missed positive labels**. Its false positive rate is `126 / 430,618 = 0.0293%`. Compared with the preceding XGBoost version, it finds **23 more positives** and creates **71 more false alerts**. Recall rises from **36.5% to 41.2%**, while precision falls from **76.4% to 61.5%**; F1 remains about **0.49**. All four classroom goals are met. The payment-format breakdown shows **201 of 428 ACH positives caught and none of 60 positive Bitcoin, Cash, Cheque, or Credit Card rows**. There were no positive Wire rows in this period. This is a substantial coverage gap despite the higher overall recall.

The notebook also shows the other validation-selected XGBoost thresholds so the tradeoff is explicit:

| Threshold policy | Later precision | Later recall | Later F1 | Later FPR | Later alerts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Best validation F1 | 73.6% | 36.1% | 0.484 | 0.0146% | 239 |
| **At least 70% validation precision (chosen)** | **61.5%** | **41.2%** | **0.493** | **0.0293%** | **327** |
| At least 40% validation precision | 35.3% | 47.5% | 0.405 | 0.0989% | 658 |

The last row finds more positive labels but doubles the chosen policy's alerts and sits close to the classroom FPR limit. None of these settings fixes non-ACH detection.

The day breakdown also matters: XGBoost recall is **39.5% on September 9** and **43.1% on September 10**, with **104** and **97** caught positives. Precision is **59.8%** and **63.4%** on those days. The two days have very different transaction volumes, so one overall rate can hide changing alert workload. These subgroup numbers are descriptive and come from small positive counts compared with the full transaction volume.

All three final models use the same **46 encoded features** and positive-class weight of 10. Logistic regression is linear; XGBoost can learn interactions; EBM uses nonlinear additive effects with interactions disabled. We show logistic coefficients, one XGBoost Tree SHAP breakdown, and EBM term importance in section 9. Explanations describe model behavior, not why IBM assigned a label. A bank would set its alert threshold using its own risk and review capacity; our precision floor is a transparent classroom choice, **not a regulatory requirement**.

## Limits and bank context

This is an exploratory comparison on sampled synthetic data. History uses all eligible earlier rows, but the fitted models and evaluation still use a 50% sample. September 11–18 is excluded, the reporting period was consulted during revisions, and model scores are **not calibrated probabilities**. No score or synthetic label is an investigator decision. The zero non-ACH recall is a specific limitation for this class report.

The CSV has synthetic laundering labels, but no investigation outcomes or SAR decisions. The notebook can calculate classifier precision and recall against those synthetic labels; it cannot measure a bank's actual SAR conversion rate or investigation quality. The [IBM dataset paper](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) notes that its complete synthetic ground truth differs from real AML data, where many laundering transactions are never detected.

Banks tailor suspicious-activity monitoring to their risks. They may combine transaction reports, rules or intelligent surveillance, and referrals; investigators then review customer and transaction context before making a Suspicious Activity Report decision. The [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04) describes this broader workflow. The notebook covers only transaction ranking. Current [interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) also discusses appropriate testing, limitations, and ongoing monitoring for bank models.

Adding relevant behavior features is normal in AML monitoring: the FFIEC manual describes systems that compare activity with account history and peers. It does **not** prescribe using every raw column or a fixed number of features. It expects monitoring criteria to match the bank's risk profile and to be reviewed and tested. The [2026 interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) calls for data and method choices aligned with model purpose, attention to input quality, and appropriate validation. The CSV has no customer-risk, occupation, geography, or due-diligence data; those cannot be honestly added from this source.
