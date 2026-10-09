# Explainable AML Transaction Monitoring

A one-notebook ML class project using **SAML-D**, a synthetic transaction-monitoring dataset. Open [aml_modeling.ipynb](aml_modeling.ipynb) for the data profile, all modeling code, saved results, charts, and conclusions. [PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) explains the steps and metrics; [CLASS_REPORT.md](docs/CLASS_REPORT.md) is the short report.

The notebook compares **logistic regression, XGBoost with Tree SHAP, and Explainable Boosting Machine (EBM)**. It uses all 9,504,852 transactions, starts with quick EDA of label balance, payment types, monthly rates, and same-currency amounts, then builds simple features from strictly earlier account activity. It checks models in expanding time windows, selects alert thresholds on validation data, and evaluates a later test period. It audits missed labels by payment type and laundering typology. All results stay in the notebook; there is no results folder, dashboard, dbt project, or test suite.

## Set up

Use Python 3.11 or newer from the repository folder:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name aml-project --display-name 'Python (AML project .venv)'
```

On macOS, XGBoost may need OpenMP (`brew install libomp`). In VS Code, select **Python (AML project .venv)** as the notebook kernel and run cells from top to bottom. The native EBM explanation graphs use Plotly and are interactive in the notebook. The full-data run needs substantial memory and time because it fits three models in three time folds and then fits each model again.

## Get the data

Download the author's [SAML-D dataset on Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml), extract `SAML-D.csv`, and place it at `data/raw/SAML-D.csv`. The author's [dataset repository](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D) describes the fields and cites the original paper. A [public SAML-D mirror](https://huggingface.co/datasets/LordNR/AMLGraphX-SAML-D) is also available; the notebook run in this repository used that mirror's archive. To fetch the same archive without Kaggle sign-in:

```bash
mkdir -p data/raw
curl -L --fail --output data/raw/SAML-D.zip 'https://huggingface.co/datasets/LordNR/AMLGraphX-SAML-D/resolve/main/SAML-D.zip?download=true'
unzip -p data/raw/SAML-D.zip SAML-D.csv > data/raw/SAML-D.csv
```

The downloaded archive used for this project has SHA-256 `3abd02c11143e395f7f0f12224001530ce612f8e3c94db4e49ec11c764ee5e96`; the extracted CSV has SHA-256 `5b71ce2ea7b47fe6f19da1aa151776b04ec74560a852c2c077df91d20b8b4ef9`. The CSV and ZIP are excluded from Git. The raw CSV is about 950 MiB.

## Read the notebook

| Sections | What happens |
| --- | --- |
| **1–2** | Load every SAML-D row and explore label balance, payment types, monthly rates, and same-currency amounts. |
| **3–4** | Calculate earlier account activity; split by time; define model inputs. |
| **5–6** | Define three explainable models and compare them in expanding time folds. |
| **7–8** | Fit on training data, choose thresholds on validation, and report later test metrics. |
| **8.1** | Check which payment methods and laundering patterns the selected model misses. |
| **9–10** | Show a separate scorecard and explanation for each model, including InterpretML's built-in EBM views, then summarize the result. |

`Laundering_type` describes a synthetic scenario and is **excluded from all model inputs** to prevent target leakage. Raw account IDs are used to calculate prior activity, then excluded from model inputs. Scores rank synthetic labels for review; they are neither calibrated crime probabilities nor evidence of actual laundering.
