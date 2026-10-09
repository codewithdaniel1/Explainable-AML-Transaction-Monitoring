# Explainable AML Modeling

A one-notebook ML class project using IBM's **synthetic HI-Small** transactions. Open [aml_modeling.ipynb](aml_modeling.ipynb) for all modeling code, saved tables, and charts. The [project guide](docs/PROJECT_GUIDE.md) explains the steps, metrics, and limits.

The notebook uses **every transaction in the 5,078,345-row CSV**. It compares logistic regression, XGBoost, and Explainable Boosting Machine (EBM). Four account-history features use only transactions before each row. Three expanding time folds compare model ranking; September 7–8 sets alert thresholds; all September 9–18 rows form the later test. The notebook reports September 9–10 and the unusually labeled September 11–18 tail separately. Each model has its own metric chart and explanation, including built-in Tree SHAP for XGBoost. The project writes no metrics JSON, predictions CSV, or `results/` folder.

## Set up and run

Use Python 3.11 or newer. From the repository folder:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name aml-project --display-name 'Python (AML project .venv)'
```

On macOS, XGBoost may need OpenMP (`brew install libomp`). In VS Code, select **Python (AML project .venv)** as the notebook kernel. Run cells from top to bottom. The full-data run fits three models repeatedly, so it takes time and needs several gigabytes of available memory. Saved outputs can be read without rerunning.

Download `HI-Small_Trans.csv` from [IBM's Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and place it at `data/raw/HI-Small_Trans.csv`. With Kaggle's CLI:

```bash
.venv/bin/python -m pip install kaggle
mkdir -p data/raw
.venv/bin/kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw --unzip
```

Kaggle may require sign-in. The raw CSV is about 454 MiB and is excluded from Git. The saved run used the file with SHA-256 `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`; check with `shasum -a 256 data/raw/HI-Small_Trans.csv` if you need the same results.

## Read the notebook

| Sections | What you will see |
| --- | --- |
| **1–3** | Load the full CSV and create four strictly earlier account-history features. |
| **4** | Split all rows by date: September 1–6 training, September 7–8 validation, and September 9–18 later test. |
| **5–6** | Define three explainable models and compare them using expanding time folds inside training. |
| **7–8** | Fit final models, choose alert thresholds on validation, and show metrics for the full test and its two distinct date ranges. |
| **9** | A separate metric chart and global and local explanation for each model. |
| **10** | Limits of the synthetic experiment. |

All 11 raw CSV fields are read. Bank and account IDs make relationship features and same-bank/account flags, then are dropped as raw identifiers. The notebook's scores rank transactions for review; they are not calibrated probabilities or proof of laundering. The later test was consulted during development, so its results are exploratory. See the [guide](docs/PROJECT_GUIDE.md) for the field audit, label balance, and interpretation.

IBM describes the source as simulated transactions for AML research. See [IBM's AML-Data repository](https://github.com/IBM/AML-Data) and the Kaggle distribution for source and license details.
