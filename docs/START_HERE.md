# Start here: reading the AML project

This guide follows the introductory pipeline from input to evaluation. Read it beside the code or run the [walkthrough notebook](../notebooks/aml_pipeline_walkthrough.ipynb). The notebook's model-training example uses **artificial demo data**; the separate [IBM study](STUDY_RESULTS.md) evaluates a 10% sample with complete earlier transaction history.

For setup, data download, and commands, use the [README](../README.md). The notebook has saved outputs; a fresh clone must regenerate the ignored IBM profile and study files to rerun those sections.

The [MVP scope](PROJECT_SCOPE.md) defines what belongs in the class project and what remains optional.

## The big picture

```text
IBM CSV or artificial demo (introductory CLI)
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
terminal/notebook   display the demo comparison and ranked examples
```

`cli.py` is the conductor: it calls those modules in order. The command `.venv/bin/aml ...` starts in `cli.py` at `main()`, which parses command options and calls `run()`.

The real-data study has a separate path: `study.py` samples modeling rows, `full_history.py` calculates features from **all** earlier IBM transactions, and `diagnostics.py` analyzes held-out errors. `replay.py` is an optional extension.

## Read these files in this order

| Order | File | What to look for |
| ---: | --- | --- |
| 1 | [README.md](../README.md) | The research question, setup commands, and current limits. |
| 2 | [data_profile.md](data_profile.md) | What the full IBM dataset actually contains and the late-period label shift. |
| 3 | [STUDY_RESULTS.md](STUDY_RESULTS.md) | The sampled real-data design, results, and limits. |
| 4 | [cli.py](../src/aml_monitoring/cli.py) | Read `main()` and `run()` to see how the introductory path connects the modules. |
| 5 | [data.py](../src/aml_monitoring/data.py) | CSV checks, sorting, and the artificial demo generator. |
| 6 | [evaluation.py](../src/aml_monitoring/evaluation.py) | Read `chronological_split()` first; return to the metrics after reading models. |
| 7 | [features.py](../src/aml_monitoring/features.py) | Earlier sender activity and same-time exclusion. |
| 8 | [models.py](../src/aml_monitoring/models.py) | Three ML models and the introductory fixed rules. |
| 9 | [study.py](../src/aml_monitoring/study.py) and [full_history.py](../src/aml_monitoring/full_history.py) | Sampled model comparison with complete earlier history. |
| 10 | [diagnostics.py](../src/aml_monitoring/diagnostics.py) | Held-out errors and uncertainty. |
| 11 | [tests](../tests) | Small examples that check time safety, currencies, and alert counting. |

## Follow the actual execution, step by step

### 1. Read and check transactions

`load_ibm_csv()` expects IBM's 11 CSV columns. The original file has two columns named `Account`; the loader gives them separate names, `from_account` and `to_account`. It keeps bank and account IDs as text, checks that labels are 0 or 1, checks amounts and timestamps, and sorts rows chronologically.

The **label** is `is_laundering`. It is the value we want the model to predict. The other transaction fields are potential inputs. Account IDs help build history but are not fed directly to the models.

### 2. Divide time into three periods

The introductory CLI's `chronological_split()` targets 60% of rows for training, 20% for validation, and 20% for test, keeping equal timestamps together. The [IBM study](STUDY_RESULTS.md) instead fixes calendar windows: September 1–6 for training, 7–8 for validation, and 9–10 for its primary test. September 11–18 is a separate stress period.

The [full-data profile](data_profile.md) shows the label shift that motivated those study windows.

### 3. Build features available at scoring time

`build_features()` turns raw fields into model inputs. Some describe the current transaction, such as hour and payment format. Others describe the sender's **earlier** activity, such as prior outgoing count over 24 hours or prior average amount over 7 days.

Example: if an account has no prior activity, sends two transactions at 10:00, and sends another at 11:00, the two 10:00 transactions see zero earlier transactions. The 11:00 transaction sees both. This prevents same-time CSV row order from creating artificial history.

Current rolling amount features group by payment currency. Sender transaction counts and new-counterparty history span currencies. The sampled IBM study uses DuckDB to compute these histories from **all** original transactions before selecting model rows.

### 4. Fit models on training rows

`fit_model()` trains logistic regression, XGBoost, or EBM. Within each study feature set, the models use the same columns. Logistic regression and XGBoost encode categorical values; EBM reads the named columns directly. The initial EBM uses main effects without learned interactions. Class weighting reflects how rare laundering labels are in training.

`rule_scores()` supplies the original **demo-only** fixed comparison using amount, 24-hour transaction count, and amount relative to prior history. The sampled study instead uses training-only, currency-specific 99th-percentile amount cutoffs in `study.py`.

### 5. Score validation and test rows

The fitted models return scores between 0 and 1 for validation and test transactions. Higher scores rank a transaction higher for review. The scores have not been calibrated, so do not read `0.8` as an 80% real-world laundering probability.

`score_predictions()` calculates:

- **Average precision:** a summary of the precision–recall curve, stored under the current JSON key `pr_auc`.
- **ROC-AUC:** ranking quality across positive and negative classes.
- **Precision@K:** positive labels among the top `K` scored alerts.
- **Recall@K:** share of all positive labels found within those top `K` alerts.
- **Brier score:** probability error; currently diagnostic because weighted model outputs are uncalibrated.

For example, if the top 100 alerts contain 20 labeled transactions out of 50 total positives, Precision@100 is 20% and Recall@100 is 40%.

### 6. Inspect the results

The demo command prints a brief comparison in the terminal. The notebook displays the full demo metrics table and highest-ranked example transactions without writing duplicate JSON or CSV files. The full IBM profile still saves `data_profile.json` so the notebook can read the 5-million-row summary without loading the CSV again.

We have run the **full IBM data profile** and a **10% sampled model study with complete earlier history**. The artificial demo is a code check. The sampled study saves local intermediate files for diagnostics; the notebook displays its model comparison, top alerts, and explanations. For reproduction commands, return to the [README](../README.md); for research interpretation, read the [study report](STUDY_RESULTS.md).
