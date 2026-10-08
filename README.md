# Explainable AML Transaction Monitoring

**New to this repository?** Read [Start here: reading the AML project](docs/START_HERE.md) for the file order and a step-by-step walkthrough of the pipeline.

See [Project scope and build status](docs/PROJECT_SCOPE.md) for all 14 planned sections and what is actually implemented.

Prefer running each stage yourself? Open the [pipeline walkthrough notebook](notebooks/aml_pipeline_walkthrough.ipynb) in VS Code and select the project's `.venv` Python kernel. It has saved outputs for the full IBM profile and for every stage of the artificial demo.

If VS Code shows another kernel such as `NowEDA (.venv311)`, click the kernel name at the top right, choose **Select Another Kernel → Python Environments**, and select this project's `.venv/bin/python`. The project's environment already includes XGBoost and EBM.

A research project comparing a fixed rule baseline, logistic regression, XGBoost, and an Explainable Boosting Machine (EBM) on **synthetic** transaction data. The main study will use IBM's `HI-Small_Trans.csv`. The code already provides a local first pass: ingest, point-in-time behavioral features, chronological evaluation, and alert-capacity metrics.

The model ranks transactions for further review. IBM's synthetic laundering label is neither a real suspicious activity report nor evidence that a real person committed a crime.

## Run the first working slice

Use Python 3.11 or newer:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,notebook]'
.venv/bin/aml --demo --output results/demo
.venv/bin/python -m pytest
```

On macOS, XGBoost may require `brew install libomp`. The `--demo` data is generated locally with an intentionally easy artificial label pattern. Its metrics only prove that the pipeline runs; **do not cite them as AML findings**.

The IBM `HI-Small_Trans.csv` file is already downloaded locally in `data/raw/` from the [IBM dataset page on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). To reproduce its profile, run:

```bash
.venv/bin/aml --csv data/raw/HI-Small_Trans.csv --profile-only --output results/ibm
```

The full-data model command is documented in the walkthrough for use after we resolve the late-period label shift and currency treatment.

The full CSV is large. The current pandas implementation loads it into memory and builds features in one process; a machine with limited RAM may need a chunked or Databricks path. `--max-rows` is available for development, but it takes a prefix of the file and its metrics cannot stand in for the full out-of-time experiment.

The first full-data profile and its validation findings are in [docs/data_profile.md](docs/data_profile.md). Review its late-period label shift before running or interpreting the model comparison.

Outputs in the ignored `results/` folder:

- `data_profile.json`: row counts, label prevalence, date range, missing values, payment formats and currencies, amount quantiles, daily activity, and split diagnostics.
- `metrics.json`: PR-AUC, ROC-AUC, Brier score, Precision@K, Recall@K, and alerts reviewed.
- `predictions.csv`: validation and test scores with source row, timestamp, and label for later error analysis.

The models train only on the earliest period. The next period is reserved for selection and threshold work, and the latest period is the final holdout. The current run reports validation and test metrics without tuning. Weighted training scores have **not** been calibrated, so Brier scores are diagnostic and model outputs should not yet be interpreted as true laundering probabilities.

## Current research design

| Question | Current implementation |
| --- | --- |
| Can models rank labeled transactions for review? | Four baselines, PR-AUC, Precision@K, Recall@K |
| Can behavior help beyond transaction fields? | Prior outgoing count and amount over 24 hours and 7 days, historical average, new counterparty indicator |
| Can evaluation avoid future-data leakage? | Features use strictly earlier timestamps; equal-time transactions do not see one another; chronological splits |
| How do interpretable models compare? | Logistic regression and EBM are trained; detailed explanation exports are next |

`amount_paid` is measured in the sending currency. A rolling sum or average across mixed payment currencies is not financially comparable. The current behavioral amount features are exploratory; currency-specific histories or currency conversion will be added before drawing substantive conclusions from them. The fixed rule threshold of 10,000 is likewise illustrative and currency dependent.

## Build sequence

1. **Core class study:** inspect the full IBM data; add currency-consistent history, transaction-only versus behavioral feature comparisons, validation-only model selection, SHAP for XGBoost, EBM term explanations, error analysis, and a concise research report.
2. **Engineering:** add dbt transformations and a Databricks path, MLflow tracking, reproducible runs, and monitoring/backtests where the observed time range supports them.
3. **Extensions:** graph features and possible GNN/MLP benchmarks, replayed transaction scoring, an investigator dashboard, and synthetic investigation notes. These remain separate from the primary three-model comparison.

No real transaction data, credentials, or model artifacts should be committed. `data/raw/`, `results/`, `models/`, and `.env` are ignored.

## Source

IBM describes the data as generated by a simulated economy, with a laundering tag for model research. See [IBM's AML-Data repository](https://github.com/IBM/AML-Data) and its linked [Kaggle distribution](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). IBM notes that the data has a separate CDLA-Sharing-1.0 license; check its terms before redistribution.
