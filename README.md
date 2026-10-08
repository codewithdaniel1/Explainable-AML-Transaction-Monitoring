# Explainable AML Modeling

A one-notebook ML class project using IBM's **synthetic HI-Small** transaction data. Open [aml_modeling.ipynb](aml_modeling.ipynb) to run the workflow. Read the short [project guide](docs/PROJECT_GUIDE.md) for an explanation of the steps, features, models, results, and limits.

The notebook holds the code, tables, and charts. It saves outputs **inside the notebook**, with no `results/` folder, metrics JSON, predictions CSV, separate Python modules, app, or test suite. The raw dataset remains a local file because it is about 454 MiB and is not included in Git.

## What runs today

```text
IBM HI-Small CSV → full-data profile and 10% modeling sample
                 → DuckDB features from every earlier source transaction
                 → rules, logistic regression, XGBoost, and EBM
                 → validation selection, holdout metrics, explanations, case review
```

The notebook's numbered sections follow this order. The [project guide](docs/PROJECT_GUIDE.md) explains the purpose of each step, the feature and model choices, how to read the metrics, and the study's limitations. The saved notebook outputs can be read without downloading the CSV; rerunning the cells needs the raw file and can take several minutes.

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

For the exact source version used for the saved results, `shasum -a 256 data/raw/HI-Small_Trans.csv` should report `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`. A different file or row order can change the sampled rows and metrics.

## Study boundary

Training uses September 1–6, validation September 7–8, and the primary holdout September 9–10. September 11–18 has a sharp volume and label-rate shift, so it is a separate stress check. The notebook selects the model using validation average precision and compares rankings at a fixed 100-alert review capacity. Weighted scores are not calibrated probabilities, and synthetic labels do not establish real-world AML effectiveness.

## Databricks and dbt status

The original proposal included Databricks and dbt as a possible engineering layer. **They are not implemented in this local MVP.** DuckDB performs the earlier-history calculations here. If cloud data engineering becomes a course requirement, the [guide's Databricks/dbt section](docs/PROJECT_GUIDE.md#databricks-and-dbt-current-status-and-a-concrete-later-path) outlines how to load the raw CSV, transform and test it with dbt, and verify that its features match the notebook before using them for modeling.

IBM describes the source as simulated financial transactions for AML research. See [IBM's AML-Data repository](https://github.com/IBM/AML-Data) and the linked Kaggle distribution for source and license details. The CSV and temporary files are excluded from Git.
