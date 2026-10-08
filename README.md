# Explainable AML Transaction Monitoring

Research code for ranking **synthetic** transactions for AML review. The planned class study compares fixed rules, logistic regression, XGBoost, and an Explainable Boosting Machine (EBM) on IBM's HI-Small dataset. A model score recommends review; the synthetic laundering label is not a real suspicious activity report or a finding about a person.

## What works now

- The IBM `HI-Small_Trans.csv` was downloaded and profiled in the original workspace. The [profile report](docs/data_profile.md) records 5,078,345 transactions and a major late-period label shift.
- The pipeline loads IBM-format CSVs, builds features from earlier transaction history, and makes chronological train/validation/test splits.
- Fixed rules, logistic regression, XGBoost, and EBM run on a **small artificial demo**. The [walkthrough notebook](notebooks/aml_pipeline_walkthrough.ipynb) has saved outputs for each step and a snapshot of the IBM profile.
- **No model performance on the full IBM dataset has been reported.** Currency-mixed amount features and the late-period label shift must be addressed first.

For a file-by-file explanation, read [Start here](docs/START_HERE.md). The [scope map](docs/PROJECT_SCOPE.md) tracks all 14 planned sections and their actual status.

## Set up a fresh clone

These commands use macOS/Linux paths and Python 3.11 or newer:

```bash
git clone https://github.com/codewithdaniel1/Explainable-AML-Transaction-Monitoring.git
cd Explainable-AML-Transaction-Monitoring
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,notebook]'
.venv/bin/aml --demo --output results/demo
.venv/bin/python -m pytest -q
```

On macOS, XGBoost may need the OpenMP runtime (`brew install libomp`). The artificial demo has an intentionally easy label pattern; its metrics verify execution and are **not** AML research findings.

In VS Code, open [the notebook](notebooks/aml_pipeline_walkthrough.ipynb) and select this repository's `.venv/bin/python` kernel. If it does not appear, register it and reopen the kernel picker:

```bash
.venv/bin/python -m ipykernel install --user --name aml-project --display-name 'Python (AML project .venv)'
```

The notebook's first code cell checks that the selected kernel belongs to this project's `.venv`. Run cells from top to bottom. Its saved IBM tables are a snapshot; a fresh **Run All** needs the generated profile JSON described below to display those tables.

## Download and profile the IBM data

The raw CSV and generated `results/` files are **not in this GitHub repository**. Download only the HI-Small transaction file from [IBM's Kaggle distribution](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml):

```bash
.venv/bin/python -m pip install kaggle
mkdir -p data/raw
.venv/bin/kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw --unzip
.venv/bin/aml --csv data/raw/HI-Small_Trans.csv --profile-only --output results/ibm
```

Kaggle may ask you to sign in. The extracted CSV is about 454 MiB; verify its SHA-256 against the value in the [profile report](docs/data_profile.md). Profiling reads and sorts all 5 million rows in memory. `--max-rows` reads only a development prefix and cannot provide valid full-data evaluation results.

The profile command writes `results/ibm/data_profile.json`. Open it for exact counts, dates, daily activity, currencies, payment formats, missing values, amount quantiles, and split diagnostics. Then rerun the notebook's profile cell to display its IBM tables.

## Current pipeline and outputs

The CLI (`src/aml_monitoring/cli.py`) connects CSV loading, feature engineering, time splitting, models, and evaluation. Without `--profile-only`, it writes the following under the ignored output directory:

| File | Contents |
| --- | --- |
| `data_profile.json` | Dataset and split summaries. |
| `metrics.json` | Average precision (stored as `pr_auc`), ROC-AUC, Brier score, Precision@K, and Recall@K for validation and test. |
| `predictions.csv` | Validation and test scores, labels, timestamps, model names, and row positions. |

The models fit on the earliest period only. The current code reports validation and test metrics without tuning or probability calibration. Weighted model scores should not be read as real-world laundering probabilities. The full-data modeling path is still a research work item; the current pandas feature builder may require substantial RAM on 5 million rows.

## Next research work

1. Investigate the IBM dataset's sharp volume and label-rate change after September 10, then document the final out-of-time holdout design.
2. Make historical amount features and rules currency-consistent.
3. Compare transaction-only features with behavioral features; tune on validation data, then evaluate the untouched holdout.
4. Add SHAP, EBM explanations, error analysis, and a class research report.

Databricks, dbt, MLflow, deep learning, graph analysis, streaming, and an investigation dashboard remain separate planned sections in the [scope map](docs/PROJECT_SCOPE.md).

## Data source and repository hygiene

IBM describes this dataset as transactions generated by a simulated economy with laundering labels for model research. See [IBM's AML-Data repository](https://github.com/IBM/AML-Data) and its linked [Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). IBM states that the data has a separate CDLA-Sharing-1.0 license. The raw CSV is not redistributed here.

`.gitignore` excludes `data/raw/`, `results/`, `models/`, `.venv/`, and `.env` so large data, generated outputs, and local credentials stay out of Git.
