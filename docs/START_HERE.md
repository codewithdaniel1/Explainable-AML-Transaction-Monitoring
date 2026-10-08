# Start here: reading the AML project

This guide follows one transaction from input to model evaluation. You can read it alongside the code in your IDE. The runnable model example uses **artificial demo data**; the full IBM dataset has been profiled but has not been used for a reported model comparison.

To run the steps as notebook cells with visible outputs, follow the [README setup instructions](../README.md), open [the pipeline walkthrough notebook](../notebooks/aml_pipeline_walkthrough.ipynb), and select this repository's `.venv/bin/python` kernel. The notebook contains a saved IBM profile snapshot. A fresh clone needs the separate download and profile commands from the README to regenerate those tables.

The [project scope map](PROJECT_SCOPE.md) lists all 14 sections we discussed and marks which parts are working, partial, or planned.

## The big picture

```text
IBM CSV or artificial demo
        ↓
data.py             load, check, sort transactions
        ↓
evaluation.py       choose train, validation, and test periods
        ├─────────── profile-only: save data_profile.json and stop
        ↓ full run
features.py         calculate what was known at each transaction time
        ↓
models.py           score fixed rules; fit logistic regression / XGBoost / EBM
        ↓
evaluation.py       calculate average precision and alert-capacity metrics
        ↓
results/            save the profile, metrics, and predictions
```

`cli.py` is the conductor: it calls those modules in order. The command `.venv/bin/aml ...` starts in `cli.py` at `main()`, which parses command options and calls `run()`.

## Read these files in this order

| Order | File | What to look for |
| ---: | --- | --- |
| 1 | [README.md](../README.md) | The research question, setup commands, and current limits. |
| 2 | [data_profile.md](data_profile.md) | What the full IBM dataset actually contains and the late-period label shift. |
| 3 | [cli.py](../src/aml_monitoring/cli.py) | Read `main()` first, then `run()`. Notice where each other module is called. |
| 4 | [data.py](../src/aml_monitoring/data.py) | `load_ibm_csv()` reads the file; `normalize_transactions()` checks types and sorts by time. `make_demo_transactions()` is only for a smoke test. |
| 5 | [evaluation.py](../src/aml_monitoring/evaluation.py) | Read `chronological_split()` now. Come back to `top_k_metrics()` and `score_predictions()` after reading the models. |
| 6 | [features.py](../src/aml_monitoring/features.py) | `build_features()` uses `_SenderHistory` to calculate 24-hour and 7-day history from earlier transactions. |
| 7 | [models.py](../src/aml_monitoring/models.py) | `fit_model()` builds the three ML models; `rule_scores()` is the fixed rule baseline. |
| 8 | [evaluation.py](../src/aml_monitoring/evaluation.py) | Finish with the metric functions. |
| 9 | [test_pipeline.py](../tests/test_pipeline.py) | See the small examples that check CSV parsing, time leakage, splits, and alert counting. |

## Follow the actual execution, step by step

### 1. Read and check transactions

`load_ibm_csv()` expects IBM's 11 CSV columns. The original file has two columns named `Account`; the loader gives them separate names, `from_account` and `to_account`. It keeps bank and account IDs as text, checks that labels are 0 or 1, checks amounts and timestamps, and sorts rows chronologically.

The **label** is `is_laundering`. It is the value we want the model to predict. The other transaction fields are potential inputs. Account IDs help build history but are not fed directly to the models.

### 2. Divide time into three periods

`chronological_split()` targets 60% of rows for training, 20% for validation, and 20% for the final test. It keeps transactions with the same timestamp together. Training uses the earliest period; the latest period is reserved for out-of-time evaluation.

For the real IBM file, see the exact dates and label counts in [data_profile.md](data_profile.md). The sharp label shift near the end needs investigation before finalizing a research-grade holdout.

### 3. Build features available at scoring time

`build_features()` turns raw fields into model inputs. Some describe the current transaction, such as hour and payment format. Others describe the sender's **earlier** activity, such as prior outgoing count over 24 hours or prior average amount over 7 days.

Example: if an account has no prior activity, sends two transactions at 10:00, and sends another at 11:00, the two 10:00 transactions see zero earlier transactions. The 11:00 transaction sees both. This prevents same-time CSV row order from creating artificial history.

The current amount-history features mix source currencies. Treat them as exploratory until we group histories by currency or convert amounts consistently.

### 4. Fit models on training rows

`fit_model()` trains logistic regression, XGBoost, or EBM. Each gets the same feature columns. Logistic regression and XGBoost encode categorical values; EBM reads the named columns directly. The initial EBM uses main effects without learned interactions. Class weighting reflects how rare laundering labels are in the training period.

`rule_scores()` supplies a simple fixed comparison using amount, 24-hour transaction count, and amount relative to prior history. Its thresholds are illustrative and need currency-specific revision.

### 5. Score validation and test rows

The fitted models return scores between 0 and 1 for validation and test transactions. Higher scores rank a transaction higher for review. The scores have not been calibrated, so do not read `0.8` as an 80% real-world laundering probability.

`score_predictions()` calculates:

- **Average precision:** a summary of the precision–recall curve, stored under the current JSON key `pr_auc`.
- **ROC-AUC:** ranking quality across positive and negative classes.
- **Precision@K:** positive labels among the top `K` scored alerts.
- **Recall@K:** share of all positive labels found within those top `K` alerts.
- **Brier score:** probability error; currently diagnostic because weighted model outputs are uncalibrated.

For example, if the top 100 alerts contain 20 labeled transactions out of 50 total positives, Precision@100 is 20% and Recall@100 is 40%.

### 6. Inspect saved outputs

The command writes to the ignored `results/` folder. These generated files are not in the public GitHub repository:

| File | What it answers |
| --- | --- |
| `data_profile.json` | How many rows, positives, currencies, formats, and missing values are there? What are the split dates and daily counts? |
| `metrics.json` | How did each selected model score on validation and test? |
| `predictions.csv` | Which transaction rows received which model scores? |

So far, we have run the **full IBM data profile** in the original workspace. The artificial demo has run all four baselines as a code check. We have not reported model performance on the full IBM file.

## Commands to try

From the project root after following the [README setup](../README.md):

```bash
# Reproduce the IBM profile after downloading the CSV.
.venv/bin/aml --csv data/raw/HI-Small_Trans.csv --profile-only --output results/ibm

# Run the small artificial demo and inspect its output files.
.venv/bin/aml --demo --output results/demo

# Run checks for the parts most likely to produce misleading results.
.venv/bin/python -m pytest -q
```

The full-data model run is pending a defensible holdout design and currency-consistent features. Its present pandas implementation may require substantial RAM for 5 million rows.

## What comes next

1. Investigate the September 11–18 label shift and set the final test window.
2. Make amount-history features currency-consistent.
3. Run a transaction-only versus behavioral-feature comparison.
4. Tune models using validation data, then evaluate the untouched holdout.
5. Add SHAP and EBM explanations and write the class research report.

Databricks, dbt, MLflow, graph methods, and the investigator dashboard are later sections of the [complete project scope](PROJECT_SCOPE.md).
