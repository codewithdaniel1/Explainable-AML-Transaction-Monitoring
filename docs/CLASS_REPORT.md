# Explainable AML Transaction Monitoring: Class Report

## Research question

Can understandable machine-learning models rank synthetic transactions for AML review while keeping alert volume manageable? This project compares logistic regression, XGBoost, and an Explainable Boosting Machine (EBM). It also asks which labeled laundering patterns the strongest model misses.

## Dataset and method

The project uses **all 9,504,852 transactions** in [SAML-D](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D), a synthetic transaction-monitoring dataset. There are **9,873 positive labels (0.1039%)**. Transactions span October 7, 2022 to August 23, 2023 and include amount, payment type, currencies, bank locations, and sender and receiver account IDs. Labels appear across the timeline but vary by payment method. For example, the positive rate is **0.0577% for ACH** and **0.6239% for Cash Deposit**. These are properties of the synthetic generator, not measured bank risk.

The notebook's quick EDA charts the imbalance, payment-type rates, and monthly rates. Within **UK-pound payments**, the median positive-label amount is about **5,091**, versus **6,047** for negative-label payments. This descriptive comparison shows why a simple high-amount rule would be incomplete; the final models use amount alongside payment and account-history information. October and August are partial months in the monthly chart.

The notebook derives hour, weekday, current log amount, currency and location match flags, and simple features from strictly earlier sender, receiver, and sender–receiver pair activity. A sender's earlier average amount uses transactions in the **same payment currency**. Raw account IDs link history but are not model inputs. The `Laundering_type` field is **excluded from every model** because it identifies a generated scenario and could reveal the answer; it is used only for a post-score coverage audit.

The split keeps complete calendar days in time order:

| Period | Dates | Rows | Row share | Positive labels | Positive rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Training | Oct 7, 2022–May 18, 2023 | 6,661,223 | 70.08% | 6,774 | 0.1017% |
| Validation | May 19–Jul 5, 2023 | 1,430,805 | 15.05% | 1,416 | 0.0990% |
| Testing | Jul 6–Aug 23, 2023 | 1,412,824 | 14.86% | 1,683 | 0.1191% |

Three expanding time folds within training select the model family by mean **average precision (AP)**. XGBoost has the highest mean fold AP (**0.8376**), versus **0.3306** for EBM and **0.0799** for logistic regression. All three final models then fit on the full training period. Each model's alert threshold maximizes F1 on validation data; no test labels set a threshold.

## Later test results

Every model is evaluated on the **same 1,412,824 later transactions and 1,683 positive labels**. Each row below uses its own validation-selected threshold.

| Model | AP | Precision | Recall | F1 | FPR | Alerts | True positives | False positives | Missed positives (of 1,683) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.1336 | 21.2% | 21.7% | 0.2146 | 0.0965% | 1,728 | 366 | 1,362 | 1,317 |
| **XGBoost** | **0.8959** | **93.8%** | **84.1%** | **0.8872** | 0.0066% | 1,509 | **1,416** | 93 | **267** |
| EBM | 0.5100 | 89.3% | 41.2% | 0.5642 | **0.0059%** | **777** | 694 | **83** | 989 |

**XGBoost** is the selected model because it ranked best across the earlier time folds, and it also has the strongest test AP, precision, recall, and F1 at its validation-selected threshold. Its 1,509 alerts comprise **1,416 positive-label transactions and 93 negative-label transactions**. EBM generates fewer alerts and slightly fewer false alerts, but catches fewer than half of test positives. Logistic regression is the easiest to describe as a weighted sum, yet its alert workload is larger and its detection is much lower on this dataset. The notebook includes a separate scorecard and explanation charts for each model: global and local Tree SHAP for XGBoost, plus InterpretML's built-in global, feature-effect, and local views for EBM.

### How to read the metrics

An **alert** is a transaction above a model's validation-selected score threshold. A **true positive** is an alert with a positive *synthetic dataset label*. A **false positive** is an alert with a negative label. A **missed positive** (false negative) is a positive-label transaction below the threshold. These terms do not mean confirmed criminal activity.

- **AP** summarizes ranking across thresholds. XGBoost's **0.8959 AP** is not the percentage of its actual alerts with positive labels. The test positive-label rate, **0.1191%**, is approximately the AP baseline for random ranking.
- **Precision** = true positives ÷ alerts. For XGBoost, **1,416 ÷ 1,509 = 93.8%** of alerts match positive labels.
- **Recall** = true positives ÷ all positive labels. For XGBoost, **1,416 ÷ 1,683 = 84.1%** caught, leaving **267 missed**.
- **F1** = `2 × precision × recall ÷ (precision + recall)`. XGBoost's **0.8872** balances its precision and recall at this threshold.
- **False positive rate (FPR)** = false positives ÷ all negative labels. XGBoost produces **93 false alerts among 1,411,141 negative-label transactions**, or **0.0066% FPR**. Its small percentage should always be read with the alert count.
- **Alerts** describe transaction-level review volume. They are not necessarily unique customers or investigation cases.

## Coverage gap: which positives were missed?

XGBoost's **267 missed positive labels** are concentrated in several synthetic patterns:

| Payment type | Positive labels | Caught | Missed | Recall |
| --- | ---: | ---: | ---: | ---: |
| Cash Deposit | 232 | 117 | **115** | 50.4% |
| Cash Withdrawal | 237 | 157 | **80** | 66.2% |
| Cross-border | 477 | 439 | 38 | 92.0% |
| ACH | 176 | 165 | 11 | 93.8% |
| Cheque | 165 | 155 | 10 | 93.9% |
| Debit card | 203 | 195 | 8 | 96.1% |
| Credit card | 193 | 188 | 5 | 97.4% |

The largest typology gap is **Smurfing: 36 of 151 caught, 115 missed (23.8% recall)**. The next largest is **Cash_Withdrawal: 157 of 237 caught, 80 missed (66.2% recall)**. `Laundering_type` was read only to create this table after predictions; it did not help the models score transactions. The high aggregate recall therefore still hides weak coverage of particular generated patterns.

## Limits and conclusion

The strong XGBoost result shows that shallow trees can detect many patterns generated in SAML-D. It does **not** imply that a bank could achieve 93.8% precision or 84.1% recall on real monitoring. Synthetic labels are complete and generated from rules; real-world labels, investigation outcomes, customer context, and changing behavior are different. A label-generation shortcut is also possible in synthetic benchmarks, so unusually high scores deserve scrutiny even when the explicit typology field is excluded.

This is an educational, time-ordered model comparison. It identifies XGBoost as the best of these three choices **for this dataset and evaluation design**, while pointing to cash activity and smurfing as coverage gaps. Before operational use, a bank would need independent validation, reliable real labels and case outcomes, risk coverage analysis, review-capacity planning, and monitoring over time. No universal U.S. AML regulation specifies a required precision, recall, F1, or FPR score.

The [notebook](../aml_modeling.ipynb) contains the code, saved tables, and charts. The [project guide](PROJECT_GUIDE.md) gives the detailed feature definitions and validation steps.

**Dataset citation:** Oztas et al., “Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset,” *2023 IEEE International Conference on e-Business Engineering*, DOI [10.1109/ICEBE59045.2023.00028](https://doi.org/10.1109/ICEBE59045.2023.00028). The author notes that the currently distributed dataset is an updated version of the one used in the paper.
