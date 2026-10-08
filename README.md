# Explainable AML Transaction Monitoring

Research code for ranking **synthetic** transactions for AML review. The class study compares fixed rules, logistic regression, XGBoost, and an Explainable Boosting Machine (EBM) on IBM's HI-Small dataset. A model score recommends review; the synthetic laundering label is not a real suspicious activity report or a finding about a person.

## What works now

- The IBM `HI-Small_Trans.csv` was downloaded and profiled in the original workspace. The [profile report](docs/data_profile.md) records 5,078,345 transactions and a major late-period label shift.
- The pipeline loads IBM-format CSVs, builds features from earlier transaction history, and makes chronological train/validation/test splits.
- Fixed rules, logistic regression, XGBoost, and EBM run on a **small artificial demo**. The [walkthrough notebook](notebooks/aml_pipeline_walkthrough.ipynb) has saved outputs for each step and a snapshot of the IBM profile.
- A [10% sampled IBM pilot](docs/STUDY_RESULTS.md) now compares seven rule/model and feature-set specifications with fixed calendar splits, separate late-period stress testing, Tree SHAP examples, diagnostics, and plots. DuckDB computes behavioral history from **all** original rows before sampled modeling rows are selected.
- **No full-dataset model performance has been reported.** The pilot evaluates a sampled primary period with complete prior history, not every primary transaction.

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

The notebook's first code cell checks that the selected kernel belongs to this project's `.venv`. Run cells from top to bottom. Its saved IBM tables are a snapshot; a fresh **Run All** needs the generated profile JSON described below to display those tables. The later sampled-study section needs its separate study and diagnostics commands.

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

The demo models fit on the earliest period only. The demo reports validation and test metrics without tuning or probability calibration. The sampled IBM study uses fixed calendar windows and chooses a specification by validation average precision. Weighted scores should not be read as real-world laundering probabilities. The full-data modeling path remains a research work item; the pandas feature builder may require substantial RAM on 5 million rows.

## Run the sampled IBM pilot

After downloading the CSV, these commands reproduce the [reported study](docs/STUDY_RESULTS.md):

```bash
.venv/bin/python -m pip install -e '.[full-history,visualize]'
.venv/bin/python -m aml_monitoring.study --csv data/raw/HI-Small_Trans.csv --fraction 0.10 --seed 42 --full-history --output results/study_full_history_10pct
.venv/bin/python -m aml_monitoring.diagnostics --study-dir results/study_full_history_10pct --repeats 300
.venv/bin/python -m aml_monitoring.replay --study-dir results/study_full_history_10pct --daily-capacity 100
```

The reported run uses about 507,000 sampled September 1–10 rows plus all 1,108 late-period rows. It writes metrics, predictions, explanations, diagnostics, and a saved-score alert replay under ignored `results/study_full_history_10pct/`. The [study report](docs/STUDY_RESULTS.md) explains the design and limits. Regenerate its figures with `.venv/bin/python -m aml_monitoring.plots`.

To review the generated synthetic alerts in a local dashboard:

```bash
.venv/bin/python -m pip install -e '.[dashboard]'
.venv/bin/streamlit run app/dashboard.py
```

The dashboard compares specifications, shows ranked transactions, case context, a sender's earlier seven days and counterparties, and Tree SHAP examples. It saves optional local case notes under ignored `results/`. Read the [model card](docs/MODEL_CARD.md) for intended use and validation limits.

GitHub Actions CI runs unit tests, the artificial demo, and notebook validation on pushes and pull requests. It does not need or download the IBM CSV.

## Next research work

1. Extend model training and primary evaluation from the 10% sample to all eligible transactions.
2. Investigate the dataset's sharp volume and label-rate change after September 10; retain a documented primary holdout and separate stress period.
3. Repeat experiments across seeds or windows, tune on validation data, calibrate scores, and estimate uncertainty with account and time dependence.
4. Expand explanation and case review, then complete the class research report.

Databricks, dbt, MLflow, deep learning, graph-based detection, and live streaming remain separate planned sections in the [scope map](docs/PROJECT_SCOPE.md). The local dashboard and saved-score replay are initial investigation and monitoring interfaces.

## Data source and repository hygiene

IBM describes this dataset as transactions generated by a simulated economy with laundering labels for model research. See [IBM's AML-Data repository](https://github.com/IBM/AML-Data) and its linked [Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). IBM states that the data has a separate CDLA-Sharing-1.0 license. The raw CSV is not redistributed here.

`.gitignore` excludes `data/raw/`, `results/`, `models/`, `.venv/`, and `.env` so large data, generated outputs, and local credentials stay out of Git.
