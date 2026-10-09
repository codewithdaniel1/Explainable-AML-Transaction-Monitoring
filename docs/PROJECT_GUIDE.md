# Guide to the AML modeling notebook

The [notebook](../aml_modeling.ipynb) contains all code, tables, and charts for this class MVP. The [README](../README.md) covers setup and download. The task is to **rank synthetic laundering-labeled transactions for review**. A high score is not proof of laundering.

## Read the notebook in order

| Sections | What happens | Why |
| --- | --- | --- |
| **1–3** | Read every CSV row and calculate four features from strictly earlier account activity. | Use the complete data while keeping future activity out of each row's features. |
| **4** | Split all rows by date and prepare model inputs. | Separate fitting, threshold selection, and later reporting. |
| **5–6** | Define three simple models and compare them across three expanding time folds. | Choose a model using earlier ranking results. |
| **7–8** | Fit on all training rows, set alert thresholds on validation, and report full and date-specific later results. | Show detection, workload, and the unusual final period. |
| **9** | Show each model's scorecard and global and local explanations, including Tree SHAP for XGBoost. | Explain how each score is calculated. |
| **10** | Summarize the limits of the synthetic experiment. | Keep class-project scores separate from bank decisions. |

The notebook keeps results in one place. It creates no metrics JSON, predictions CSV, or `results/` directory.

## Data and time-ordered validation

The local `data/raw/HI-Small_Trans.csv` has **5,078,345 rows, and all are modeled**. Every transaction contributes to the history of later transactions. Fixed date boundaries give this split:

| Period | Dates in 2022 | Rows | Share of all rows | Positive labels | Positive rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Training | September 1–6 | 3,248,921 | 63.98% | 2,530 | 0.0779% |
| Validation | September 7–8 | 965,524 | 19.01% | 1,036 | 0.1073% |
| Later test | September 9–18 | 863,900 | 17.01% | 1,611 | 0.1865% |

The later test contains two very different parts:

| Test dates | Rows | Positive labels | Positive rate |
| --- | ---: | ---: | ---: |
| September 9–10 | 862,792 | 956 | 0.1108% |
| September 11–18 | 1,108 | 655 | 59.1155% |

The small September 11–18 tail has an enormous label-rate shift. It is included because the project now uses every row, but its precision and average precision cannot be compared naively with September 9–10: those metrics depend on the positive-label rate. The all-test aggregate combines these different populations, so read both date-specific rows in the notebook. Monitoring uses past activity to review later activity; this is a time-based **evaluation**, not a forecasting model. The [IBM benchmark paper](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf) also uses temporal evaluation and features from past transactions.

### Label balance

Across the full CSV, **5,177 rows (0.1019%)** have a positive synthetic label and **5,073,168 (99.8981%)** have a negative label. Predicting everything negative would look accurate but catch nothing. This is why the notebook reports precision, recall, F1, false positive rate, average precision, and alert counts instead of relying on accuracy. The ordinary September 9–10 positive rate, about **0.00111**, is the random-ranking baseline for average precision in that period; the later tail's baseline is about **0.591**.

### Three expanding folds

Within September 1–6 training, each model is refitted on earlier days and checked on the next day:

| Fold | Fit dates | Check date | Positive labels on check day |
| --- | --- | --- | ---: |
| 1 | September 1–3 | September 4 | 407 |
| 2 | September 1–4 | September 5 | 471 |
| 3 | September 1–5 | September 6 | 531 |

Category encoding and logistic scaling use each fold's fitting rows. History features for any transaction use only activity **strictly before its timestamp**, including earlier rows outside a fold's fitting period; they use no labels. This simulates information that would have been observed by that time. Stratified K-fold would even out labels but could reverse time order. The notebook chooses the model family by mean **average precision (AP)** across these folds, then chooses thresholds on September 7–8 without refitting. The [2026 interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) discusses out-of-time testing as an option without mandating a particular cross-validation method.

| Model | Sep 4 AP | Sep 5 AP | Sep 6 AP | Mean AP |
| --- | ---: | ---: | ---: | ---: |
| **XGBoost** | 0.4124 | 0.4332 | 0.4886 | **0.4448** |
| EBM | 0.3868 | 0.3994 | 0.4543 | 0.4135 |
| Logistic regression | 0.2584 | 0.2798 | 0.2995 | 0.2792 |

## Inputs and simple model explanations

All **11 original CSV fields** are read. Timestamp sets the split and transaction hour. Bank and account identifiers connect sender, receiver, and pair histories and make the same-bank/account flags; raw IDs are **not direct model predictors**. Paid and received amounts become log amounts, currencies and payment format become 0/1 categories, and `Is Laundering` is used only as the training/evaluation label.

Four history inputs come from every earlier transaction: `prior_sender_count`, `prior_receiver_count`, `prior_pair_count`, and `log_prior_sender_mean_amount` (the sender's earlier typical paid amount in the same currency). Rows at the same timestamp do not count one another. These capture basic account and relationship behavior, not the richer graph and cycle patterns in the [IBM study](https://papers.nips.cc/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf).

| Model | Simple structure | How section 9 explains it |
| --- | --- | --- |
| **Logistic regression** | A weighted sum after standardization. It log-scales history counts and adds three readable behavior clues. | A coefficient chart gives overall direction; another chart breaks down one row's linear score. |
| **XGBoost** | 120 shallow trees with maximum depth 3. | Built-in **Tree SHAP** shows average absolute effects on 500 later rows and signed effects on one row. |
| **EBM** | An additive score from one effect per input, with interactions disabled. | Term importance gives the general pattern; term contributions break down one row's score. |

For logistic regression, the three added clues are `seen_sender_before`, `seen_pair_before`, and `amount_vs_sender_usual`. The last is the absolute gap between the current log payment and the sender's earlier typical log payment; it is zero if there is no earlier sender history. These simple transformations help a linear model handle very large count ranges. The three models use the same transactions and labels, with a few simple transformations for the linear model.

XGBoost's `pred_contribs=True` returns Tree SHAP feature contributions plus a bias term. They sum to the **raw model score**, not to a probability; the global chart is an illustrative 500-row sample. [XGBoost documents this output](https://xgboost.readthedocs.io/en/latest/python/python_api.html). EBM's local term scores likewise sum with its intercept to its raw score, as described by [InterpretML](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html). Each chart explains the model's calculation; none identifies a customer's intent or the true cause of a synthetic label.

## Performance and interpretation

**AP** summarizes ranking across thresholds. At a chosen alert threshold, **precision** is `TP / (TP + FP)`, **recall** is `TP / (TP + FN)`, **F1** balances those two, and **false positive rate (FPR)** is `FP / (FP + TN)`. Alert count matters because investigators review actual cases. [scikit-learn explains precision–recall curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).

Logistic regression and EBM use their best September 7–8 **F1** thresholds. XGBoost uses the threshold with the highest validation **recall** among thresholds with at least **70% validation precision**. This is a classroom choice to find more labels while keeping most alerts useful; it is not a regulatory cutoff. Changing a threshold alters alert precision and recall without changing ranking AP, as [scikit-learn explains](https://scikit-learn.org/stable/modules/classification_threshold.html). The notebook reports the selected XGBoost operating point and its validation-F1 alternative.

The **ordinary September 9–10** results are the most comparable with validation because their positive-label rates are close:

| Model | AP | Precision | Recall | F1 | FPR | Alerts | Positive labels caught |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.2827 | 37.2% | 32.9% | 0.349 | 0.062% | 847 | 315 of 956 |
| **XGBoost** | **0.4625** | 70.6% | **39.6%** | **0.508** | 0.018% | 537 | **379 of 956** |
| EBM | 0.4167 | **79.0%** | 31.2% | 0.447 | **0.009%** | **377** | 298 of 956 |

On the 1,108-row September 11–18 tail, XGBoost catches **389 of 655** positive labels in **425 alerts**. Its tail precision is **91.5%** and recall is **59.4%**. That high precision partly reflects a population with **59.1% positive labels**; it should not be interpreted as a comparable improvement over the ordinary days. Across the full September 9–18 test, the aggregate XGBoost numbers are **79.8% precision, 47.7% recall, and 0.597 F1** in **962 alerts**, combining both populations.

The selected XGBoost model catches **none of the 122 positive non-ACH payments** in the full later test. Its ACH recall is **768 of 1,489**, while non-ACH recall is zero. The format breakdown in section 8 makes this important coverage gap visible despite the overall score.

The project's **illustrative class goal** is at least **30% precision and recall**, **0.30 F1**, and **FPR below 0.1%** on the ordinary September 9–10 test days. These are **not industry or regulatory minimums**. The [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04) describes monitoring tailored to a bank's risks and investigation process. The [Wolfsberg Group monitoring statement](https://wolfsberg-group.org/resources/195/202) considers precision and recall alongside risk coverage and SAR-quality feedback. The [2026 interagency model-risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602a1.pdf) addresses testing and limitations in a risk-based way, without setting universal AML classifier percentages.

This CSV has synthetic labels but no investigator decisions or Suspicious Activity Report outcomes. A real bank would also examine missed-risk typologies, customer context, investigative capacity, and monitoring over time. The later test was consulted during development, so these results are **exploratory**, not an untouched final estimate. Scores are **not calibrated laundering probabilities**. The study supports a class comparison of three explainable scoring approaches, not an operational AML conclusion.
