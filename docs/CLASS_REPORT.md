# Explainable AML Transaction Monitoring: Class Report

## Research question

Can three understandable machine-learning models rank synthetic transactions for AML review while keeping alert volume manageable? This project compares logistic regression, XGBoost, and an Explainable Boosting Machine (EBM). The goal is to study model behavior, not to make real customer or regulatory decisions.

## Data and method

The project uses all **5,078,345** transactions in IBM's synthetic HI-Small dataset. Only **5,177 (0.1019%)** have a positive laundering label. Each row receives current-transaction features and four account-history features calculated from transactions strictly before its timestamp. Raw account IDs connect transaction histories but are not model inputs.

The time split is September 1–6 for training (**63.98%** of rows), September 7–8 for validation (**19.01%**), and September 9–18 for later testing (**17.01%**). Three expanding time folds inside training select the model family by mean average precision (AP). Validation labels set the alert thresholds; the test labels do not set them. XGBoost has the highest mean fold AP (**0.4448**, versus **0.4135** for EBM and **0.2792** for logistic regression).

## Results

The main comparison uses **September 9–10**, whose positive-label rate (**0.1108%**) is close to validation. All three models use the same 862,792 transactions and 956 positive labels in this period.

| Model | AP | Precision | Recall | F1 | False positive rate | Alerts | Missed positives (of 956) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.2827 | 37.2% | 32.9% | 0.349 | 0.062% | 847 | 641 |
| **XGBoost** | **0.4625** | 70.6% | **39.6%** | **0.508** | 0.018% | 537 | **577** |
| EBM | 0.4167 | **79.0%** | 31.2% | 0.447 | **0.009%** | **377** | 658 |

### How to read the metrics

An **alert** is a transaction whose model score exceeds the alert threshold chosen on validation data. A *true positive* is an alert with a positive dataset label; a *false positive* is an alert with a negative label. A *false negative* is a positive-label transaction that the model does not alert on. These terms compare predictions with the **synthetic dataset labels**, not confirmed criminal activity.

- **Average precision (AP)** summarizes how well a model ranks positive-label transactions above negative-label transactions across possible alert thresholds. Higher is better. For example, XGBoost's **0.4625 AP** does **not** mean that 46.25% of its actual alerts are correct; that is measured by precision. The positive-label rate of **0.1108%** on these test days is approximately the AP baseline for random ranking.
- **Precision** is `true positives / alerts`: the share of alerts with positive labels. XGBoost's **70.6%** means **379 of 537 alerts** match positive labels.
- **Recall** is `true positives / all positive labels`: the share of known positives caught. XGBoost's **39.6%** means it catches **379 of 956** positive labels and misses **577**. Higher recall means fewer missed positives, often at the cost of more alerts.
- **F1** combines precision and recall into one score: `2 × precision × recall / (precision + recall)`. XGBoost's **0.508** is the highest of the three here. F1 summarizes their balance, but the two underlying numbers should still be read separately.
- **False positive rate (FPR)** is `false positives / all negative labels`: the share of normal-label transactions incorrectly alerted. XGBoost's **0.018%** is **158 false alerts among 861,836 negative-label transactions**. A small percentage can still create substantial review work when the normal class is very large.
- **Alerts** is the total number of transactions sent for review at the chosen threshold. XGBoost generates **537 alerts: 379 true positives and 158 false positives**. This count helps describe investigator workload.

### What the three models show

**Logistic regression** catches **315 of 956** positives and misses **641**. Its **847 alerts** include **532 false positives**, so its **37.2% precision** and **0.062% FPR** mean more review work for fewer detected positives than XGBoost. Its **32.9% recall** and **0.349 F1** are also lower than XGBoost's.

**XGBoost** has the highest AP (**0.4625**), recall (**39.6%**), and F1 (**0.508**) on these days. It is the selected model because it also ranked best across the earlier time folds. Its threshold yields **379 detected positives** and **158 false alerts** in **537 total alerts**.

**EBM** is more selective at its chosen threshold: **298 of 377 alerts** match positive labels, giving the highest precision (**79.0%**) and lowest FPR (**0.009%**). It catches **298 of 956** positives (**31.2% recall**) and misses **658**. Its smaller workload comes with more missed positives than XGBoost.

All three meet the project's *illustrative classroom goals* of at least 30% precision and recall, F1 of at least 0.30, and FPR below 0.1% on these ordinary test days. These goals are **not regulatory or industry minimums**. The notebook includes global and local Tree SHAP charts to explain XGBoost's scores; the other two models have their own explanation charts.

## Investigation: the non-ACH coverage gap

All **122 positive non-ACH payments** in the later test occur on September 9–10, and XGBoost misses every one. They account for **122 of its 577 misses** in the main comparison. The other **455 misses** are ACH payments. The same gap appears on validation: XGBoost catches **0 of 146** positive non-ACH payments there, so this is not limited to one test day.

The training labels give the model much more evidence about ACH: **2,104 positive ACH payments** among 363,584 ACH rows, versus **426 positive non-ACH payments** among 2,885,337 non-ACH rows. On the later test, the model's selected score threshold is **0.3704**, while the *highest* score of any positive non-ACH payment is **0.0300**. The model scores these payments far below the threshold; rounding the threshold slightly would not fix the gap. These observations suggest that payment format and the very different label rates across formats are important, but they do not establish a single cause for every missed transaction.

To measure the cost of lowering the bar, we chose an **exploratory non-ACH cutoff of 0.0146** from validation scores to catch about 10% of positive non-ACH validation examples. It yields **15 positive alerts and 12,021 false alerts** on validation; applied to the later test, it yields **12 positive alerts and 9,907 false alerts**. That is only about **0.12% precision** among non-ACH test alerts. This diagnostic was explored after seeing the test gap and is **not a new final model or threshold**. The next modeling question is whether better transaction-history or network features can separate non-ACH positives from ordinary non-ACH payments without that alert burden.

## Limitations and conclusion

The coverage gap matters even though XGBoost's overall metrics are strongest. The final September 11–18 period also has an unusual **59.1% positive-label rate** in just 1,108 rows. Its metrics are reported separately because combining it with September 9–10 makes aggregate precision and AP hard to interpret.

This is an **exploratory result**: the later test was examined during project development, and the dataset contains synthetic labels rather than investigator findings or Suspicious Activity Report outcomes. Scores rank transactions for review; they are not calibrated probabilities or proof of laundering. The project demonstrates a transparent, time-ordered model comparison and identifies a concrete weakness that would need further evaluation before any real-world use.

The [notebook](../aml_modeling.ipynb) contains all code, saved results, and charts. The [project guide](PROJECT_GUIDE.md) gives the detailed feature definitions, metric formulas, and AML context.
