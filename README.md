# Explainable AML Transaction Monitoring

A one-notebook ML class project using **SAML-D**, a synthetic transaction-monitoring dataset.

## Which file should I read?

| File | Purpose |
| --- | --- |
| [PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | **Start here for the architecture:** how the three detection tracks feed the three models, and how alerts are compared. |
| [aml_modeling.ipynb](aml_modeling.ipynb) | Run the full analysis in order. It contains all code, tables, charts, and saved outputs. |
| [CLASS_REPORT.md](docs/CLASS_REPORT.md) | Read the findings, model metrics, coverage gaps, and limitations. |

The project uses all 9,504,852 transactions. **Rules, anomaly detection, and network fan-in/out are three sources of signals. Logistic regression, XGBoost, and EBM are three alternative classifiers that use those signals.** The notebook evaluates each track's alerts separately, then compares them with the selected classifier's alerts. See the [architecture diagram](docs/PROJECT_GUIDE.md#architecture-at-a-glance) for the exact flow.

## Set up

Use Python 3.11 or newer from the repository folder:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name aml-project --display-name 'Python (AML project .venv)'
```

On macOS, XGBoost may need OpenMP (`brew install libomp`). In VS Code, select **Python (AML project .venv)** as the notebook kernel and run cells from top to bottom. The native EBM explanation graphs use Plotly and are interactive in the notebook. The full-data run needs substantial memory and time because it scores 9.5 million rows with the anomaly model, fits three classifiers in three time folds, and then fits each classifier again.

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
| **1–4** | Load and explore SAML-D, build past-only transaction and relationship features, then split by time. |
| **5–7** | Run fixed rules, an unsupervised anomaly score, and account-network fan-in/out signals. |
| **8–10** | Combine track signals with base features and compare three models in expanding time folds. |
| **11–13** | Fit final models, select thresholds on validation, evaluate the later test, and compare standalone and consolidated track alerts. |
| **14–16** | Audit missed labels, inspect each model’s explanation (including native EBM views), and summarize the result. |

`Laundering_type` describes a synthetic scenario and is **excluded from all detection and model inputs** to prevent target leakage. Raw account IDs are used to calculate prior activity, then excluded from model inputs. The rules and network scores are simple teaching examples, not vendor software or full graph analytics. Scores rank synthetic labels for review; they are neither calibrated crime probabilities nor evidence of actual laundering.
