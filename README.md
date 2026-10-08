# Explainable AML Modeling

A one-notebook ML class project using IBM's **synthetic HI-Small** transaction data. Open [aml_modeling.ipynb](aml_modeling.ipynb) to follow the full workflow: data profile, chronological split, earlier account-behavior features, logistic regression, XGBoost with Tree SHAP, EBM, rule baseline, evaluation, alert replay, and an investigator-style case review.

The notebook is the code, report, and results view. It saves tables and charts **inside the notebook**, with no `results/` folder, metrics JSON, predictions CSV, separate Python modules, app, or test suite. The raw dataset remains a local file because it is about 454 MiB and is not included in Git.

## Run it locally

Use Python 3.11 or newer. From the repository folder:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name aml-project --display-name 'Python (AML project .venv)'
```

On macOS, XGBoost may need OpenMP (`brew install libomp`). In VS Code, open [aml_modeling.ipynb](aml_modeling.ipynb), select **Python (AML project .venv)** as the kernel, and run cells from top to bottom.

Download `HI-Small_Trans.csv` from [IBM's Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and place it at `data/raw/HI-Small_Trans.csv`. If you use Kaggle's CLI:

```bash
.venv/bin/python -m pip install kaggle
mkdir -p data/raw
.venv/bin/kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw --unzip
```

Kaggle may require sign-in. The notebook checks the expected 5,078,345 rows, scans the entire CSV, and computes complete earlier account history with DuckDB. A fresh full run can take several minutes and needs temporary disk space. The notebook uses a 10% modeling sample with seed 42; it does **not** claim full-dataset model performance. Its saved outputs let you read the results before rerunning it.

## Study boundary

Training uses September 1–6, validation September 7–8, and the primary holdout September 9–10. September 11–18 has a sharp volume and label-rate shift, so it is a separate stress check. The notebook selects the model using validation average precision and compares rankings at a fixed 100-alert review capacity. Weighted scores are not calibrated probabilities, and synthetic labels do not establish real-world AML effectiveness.

IBM describes the source as simulated financial transactions for AML research. See [IBM's AML-Data repository](https://github.com/IBM/AML-Data) and the linked Kaggle distribution for source and license details. The CSV and temporary files are excluded from Git.
