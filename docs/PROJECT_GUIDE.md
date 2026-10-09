# Guide to the SAML-D modeling notebook

The [notebook](../aml_modeling.ipynb) contains the code, data profile, model results, and charts for this class MVP. The [README](../README.md) covers setup and the download; the [class report](CLASS_REPORT.md) summarizes the findings.

## What is the modeling question?

Can three understandable models rank **synthetic transactions labeled as laundering** for review, and which labeled patterns do they miss? This is transaction-level classification. A model alert means its score crossed a chosen threshold; it is not a legal finding, an investigator decision, or a Suspicious Activity Report.

[SAML-D](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D) was designed as a synthetic transaction-monitoring dataset with several payment methods and suspicious-activity typologies. The local CSV has **9,504,852 transactions** from October 7, 2022 through August 23, 2023. **9,873 rows (0.1039%)** have `Is_laundering = 1`; the other 9,494,979 have label 0. No source fields are missing. The data remain highly imbalanced. Predicting every row as negative would appear more than 99% accurate but would detect none of the labeled activity.

### Payment-method balance before modeling

| Payment type | Rows | Positive labels | Positive rate |
| --- | ---: | ---: | ---: |
| ACH | 2,008,807 | 1,159 | 0.0577% |
| Cash Deposit | 225,206 | 1,405 | 0.6239% |
| Cash Withdrawal | 300,477 | 1,334 | 0.4440% |
| Cheque | 2,011,419 | 1,087 | 0.0540% |
| Credit card | 2,012,909 | 1,136 | 0.0564% |
| Cross-border | 933,931 | 2,628 | 0.2814% |
| Debit card | 2,012,103 | 1,124 | 0.0559% |

These rates describe this generator, not real bank payment risk. They show why one overall metric can hide poor coverage for a particular payment method.

The quick EDA also plots monthly positive-label rates. The first and last months are partial months, so their rates should be read with that limitation. Its same-currency amount table uses **UK-pound payments only**: the median amount is about **5,091 for positive labels** and **6,047 for negative labels**. Pooling nominal amounts across currencies would make that comparison harder to interpret.

## Read the notebook in order

| Section | What happens | Why |
| --- | --- | --- |
| 1–2 | Import packages, read all rows, chart label balance, payment-method rates, and monthly rates; compare median amounts within one currency. | Understand the data before fitting without mixing currency units. |
| 3 | Create current-transaction and strictly earlier account-history features. | Give models interpretable behavior clues without future information. |
| 4 | Split dates into train, validation, and test; define model inputs. | Keep threshold selection separate from the later evaluation. |
| 5–6 | Define the three models and run expanding time folds inside training. | Compare ranking on later periods without shuffling future rows into the past. |
| 7–8 | Fit final models on training data, set F1 thresholds on validation data, and evaluate later test rows. | Report detection and review workload at a defined operating point. |
| 8.1 | Inspect missed positive labels by payment type and typology. | Reveal coverage gaps concealed by aggregate metrics. |
| 9–10 | Show each model's scorecard and explanation, then a conclusion. | Explain both performance and how scores are formed. |

The notebook does not create a `results/` folder or separate metric and prediction files.

## Time split and cross-validation

The split uses complete calendar days in order. About 70% of the **days** are for training, 15% for validation, and 15% for testing. Daily transaction totals differ slightly, so row shares are approximate:

| Period | Dates | Rows | Row share | Positive labels | Positive rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | Oct 7, 2022–May 18, 2023 | 6,661,223 | 70.08% | 6,774 | 0.1017% |
| Validation | May 19–Jul 5, 2023 | 1,430,805 | 15.05% | 1,416 | 0.0990% |
| Test | Jul 6–Aug 23, 2023 | 1,412,824 | 14.86% | 1,683 | 0.1191% |

Within training, three **expanding time folds** fit on earlier days and check the next block of days. The model family is selected by the mean **average precision** (AP) across the folds. All three final models are then fitted on the full training period. Validation labels determine each model's alert threshold, selected for highest validation F1. The final test period supplies only evaluation and coverage analysis.

Stratified K-fold would deliberately distribute rare positives across folds but would mix dates. Time order is more relevant to a monitoring workflow, where future transactions are unavailable when earlier alerts are scored. The number of positive labels in each fold is displayed in the notebook so a thin fold is visible.

## Inputs and leakage control

Every raw CSV field has a defined role:

| Source fields | Use |
| --- | --- |
| `Date`, `Time` | Order activity, set the split, derive hour and weekday. |
| `Sender_account`, `Receiver_account` | Connect transactions to earlier sender, receiver, and pair activity; raw IDs are not predictors. |
| `Amount` | Current log amount, earlier average sender amount in the same payment currency, and gap from that average. |
| `Payment_currency`, `Received_currency` | Categorical inputs and same-currency flag. |
| `Sender_bank_location`, `Receiver_bank_location` | Categorical inputs and same-location flag. |
| `Payment_type` | Categorical input and later coverage audit. |
| `Is_laundering` | Binary target for fitting and evaluation only. |
| `Laundering_type` | **Post-score coverage audit only.** It identifies a generated scenario and would reveal information about the target if used as an input. |

The earlier-history features are sender transaction count, receiver transaction count, sender–receiver pair count, sender transaction count in the **same currency**, and the sender's earlier average amount in that currency. Transactions with the same timestamp do not count one another as earlier. These features use prior transaction facts, never prior labels. The notebook also uses current payment amount, hour, weekday, same-account flag, same-bank-location flag, same-currency flag, and the amount gap from the sender's prior average.

The logistic pipeline learns one-hot categories and numeric scaling from each fitting period. XGBoost and EBM use the named categorical columns directly. All three model families receive the same underlying transaction information.

## Models, metrics, and explanations

| Model | Shape | Explanation |
| --- | --- | --- |
| Logistic regression | A weighted sum of standardized numeric values and one-hot categories. | Global coefficient chart and local contribution chart. |
| XGBoost | 100 shallow decision trees, maximum depth 3. | Global and local **Tree SHAP** charts in raw score units. |
| EBM | Additive learned effects with interactions turned off. | InterpretML's built-in global importance view, learned feature-effect graph, and local explanation. |

Positive training labels receive 10 times the fitting weight of negative labels. This is a simple class-imbalance choice, not an estimate of bank investigation cost. The threshold is fitted separately on validation data.

- **Average precision (AP)** summarizes ranking over many possible thresholds. The positive-label rate is the approximate random-ranking baseline in each period.
- **Precision** = true positive alerts ÷ all alerts. It tells us the share of alerts that match positive *synthetic labels*.
- **Recall** = true positive alerts ÷ all positive labels. It tells us how many known positives are found.
- **F1** = `2 × precision × recall ÷ (precision + recall)`. It balances precision and recall at one threshold.
- **False positive rate (FPR)** = false positive alerts ÷ all negative labels. A tiny FPR can still mean many false alerts when the negative class is very large.
- **Alerts** = all scored transactions above the chosen threshold. **Missed positives** are false negatives.

All of these are standard classification measures, but no U.S. regulation sets a universal required AML precision, recall, F1, or FPR. A bank also has to consider risk coverage, alert workload, investigator findings, data quality, and changes over time. The [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04) describes monitoring tailored to an institution's risks and procedures. These synthetic results do not establish production readiness.

### Expanding-fold results

The three check windows begin January 4, February 18, and April 4, 2023. They contain **1,523**, **1,297**, and **1,505** positive labels respectively. AP is measured on the immediately later window in each fold:

| Model | Fold 1 AP | Fold 2 AP | Fold 3 AP | Mean AP |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.0724 | 0.0897 | 0.0776 | 0.0799 |
| **XGBoost** | **0.7737** | **0.8690** | **0.8700** | **0.8376** |
| EBM | 0.1771 | 0.3823 | 0.4324 | 0.3306 |

XGBoost is selected by the predeclared mean-fold-AP rule. These are unusually strong synthetic ranking results for XGBoost. They should prompt an audit of how the generator constructs labels and an explicit warning against treating them as realistic bank performance. The target and `Laundering_type` are excluded from all input columns.

## What the SAML-D run found

All three models use the same later test period with **1,683 positive labels**. Thresholds came from validation F1, so each model has a different alert count:

| Model | Test AP | Precision | Recall | F1 | FPR | Alerts | Caught | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.1336 | 21.2% | 21.7% | 0.2146 | 0.0965% | 1,728 | 366 | 1,317 |
| **XGBoost** | **0.8959** | **93.8%** | **84.1%** | **0.8872** | 0.0066% | 1,509 | **1,416** | **267** |
| EBM | 0.5100 | 89.3% | 41.2% | 0.5642 | **0.0059%** | **777** | 694 | 989 |

For the selected XGBoost model, the **267 missed** positives include **115 Cash Deposits**, **80 Cash Withdrawals**, and **38 Cross-border** transactions. The typology audit identifies **Smurfing** as the largest gap: **115 of 151** positive Smurfing rows are missed. This shows why payment-type and typology coverage should be read with aggregate recall. `Laundering_type` did not enter any model; it is used only to label the final audit table.

The XGBoost test scores are strikingly high for a rare-label task. SAML-D is generated, and its labels can reflect consistent rules or unusually clean patterns. The result is valid as a comparison of these three models on this CSV and split, but it is not a performance estimate for a live bank. See the [class report](CLASS_REPORT.md) for the per-model interpretation and limits.
