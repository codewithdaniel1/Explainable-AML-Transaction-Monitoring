# Explainable AML Modeling

A simple, one-notebook ML class project using IBM's **synthetic HI-Small** transactions. Open [aml_modeling.ipynb](aml_modeling.ipynb) for the code, saved results, and charts. Read the [project guide](docs/PROJECT_GUIDE.md) for the reasoning behind each step and model.

The notebook compares **logistic regression, XGBoost, and Explainable Boosting Machine (EBM)**. It fits models on a 50% sample of transactions before September 11 while calculating account-history features from **all earlier eligible transactions**. It also checks short-window activity features, uses three expanding time folds to compare models, and evaluates with average precision, precision, recall, F1, false positive rate, alert counts, and a confusion matrix. It writes no metrics JSON, predictions CSV, or `results/` folder.

## Set up and run

Use Python 3.11 or newer. From the repository folder:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name aml-project --display-name 'Python (AML project .venv)'
```

On macOS, XGBoost may need OpenMP (`brew install libomp`). In VS Code, select **Python (AML project .venv)** as the notebook kernel. Run cells from top to bottom; the three time folds fit each model repeatedly, so a full run can take several minutes. The saved outputs can be read without rerunning the notebook.

Download `HI-Small_Trans.csv` from [IBM's Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and place it at `data/raw/HI-Small_Trans.csv`. With Kaggle's CLI:

```bash
.venv/bin/python -m pip install kaggle
mkdir -p data/raw
.venv/bin/kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw --unzip
```

Kaggle may require sign-in. The raw CSV is about 454 MiB and is excluded from Git. The saved run used the file with SHA-256 `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`; check with `shasum -a 256 data/raw/HI-Small_Trans.csv` if you need the same sample and results.

## Read the notebook

| Sections | What you will see |
| --- | --- |
| **1–3** | Imports, raw-versus-sampled label balance, current-transaction fields, complete earlier sender/receiver/pair history, and candidate short-window activity features. |
| **4** | September 1–6 training (**64.02%**), September 7–8 validation (**19.00%**), and September 9–10 later test (**16.98%**) of the modeled sample. |
| **5–6** | Define the three models and compare them using expanding time folds inside the training period. |
| **7–8** | Fit final models on September 1–6, choose alert thresholds using September 7–8, compare XGBoost recall and alert workload, report September 9–10 results, and inspect misses by format and day. |
| **9** | A separate table, precision/recall/F1/FPR chart, and explanation for each model, plus feature explanations. |
| **10** | Limits of the experiment and bank AML context. |

Every raw CSV field is read. Bank and account IDs remain in the working transaction table and contribute to history and relationship features; they are not fed to the models as arbitrary identifiers. History features use all eligible earlier rows, including those outside the 50% modeling sample. The notebook excludes the unusual September 11–18 period, whose transaction volume and label rate shift sharply. Model scores are not calibrated laundering probabilities, and synthetic labels are not real AML decisions. We examined the reporting period while improving this MVP, so its metrics are exploratory. See the [guide](docs/PROJECT_GUIDE.md) for the field audit and regulatory context.

Label balance is highly uneven: **5,177 of 5,078,345 raw rows (0.1019%)** and **2,271 of 2,539,464 modeled rows (0.0894%)** have a positive synthetic label. The [guide](docs/PROJECT_GUIDE.md#label-balance) also shows the negative rates and each time split's positive rate.

The class-project goal is **at least 30% precision and recall, F1 of at least 0.30, and false positive rate below 0.1%** on September 9–10. The recall-focused XGBoost setting met all four: **61.5% precision, 41.2% recall, 0.493 F1, and 0.029% false positive rate**. Its 327 alerts caught **201 of 488** positive labels, but **none of the 60 positive non-ACH payments**. The notebook shows more conservative and more aggressive validation-selected thresholds so the recall and alert-workload tradeoff is visible. These are illustrative targets, not regulatory minimums; the [guide](docs/PROJECT_GUIDE.md#what-performance-should-this-project-aim-for) explains the limits and how bank AML goals are set.

IBM describes the source as simulated transactions for AML research. See [IBM's AML-Data repository](https://github.com/IBM/AML-Data) and the linked Kaggle distribution for source and license details.
