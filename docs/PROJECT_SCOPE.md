# Project scope and build status

Status as of 2026-10-08. The [walkthrough notebook](../notebooks/aml_pipeline_walkthrough.ipynb) teaches the pipeline on artificial data and shows saved real-data outputs. The [IBM study](STUDY_RESULTS.md) is a **10% sampled pilot** with complete earlier history. Raw data and generated results are excluded from Git; see the [README](../README.md) to reproduce them.

| Section | Component | Current status | Next deliverable |
| ---: | --- | --- | --- |
| 1 | Dataset and exploratory analysis | **Partial:** full IBM HI-Small CSV profiled; daily volume/label plot and sampled study completed | Investigate the late-period generation shift and deepen segment analysis |
| 2 | Databricks and dbt data engineering | **Planned** | Ingestion, bronze/silver/gold tables, dbt transformations and tests |
| 3 | Behavioral AML features | **Partial:** prior 24-hour and 7-day sender history from all original rows, currency-consistent amounts, and a sampled feature-set comparison | Add more behavior and scale model fitting to full data |
| 4 | Rule-based monitoring | **Partial:** demo rules and sampled-study training-only currency thresholds | Validate operational rule designs and alert volume |
| 5 | Supervised ML | **Partial:** logistic regression, XGBoost, and main-effects EBM compared on sampled IBM rows with complete prior history | Full-data evaluation and validation-only tuning |
| 6 | Deep learning | **Planned** | Separate MLP benchmark |
| 7 | Graph-based detection | **Planned** | Time-safe account graph features; possible GNN benchmark |
| 8 | Model performance metrics | **Partial:** AP, ROC-AUC, Brier, Precision@K, Recall@K, alert errors, and descriptive bootstrap interval on the sampled pilot | Account/time-aware uncertainty, computational cost, and calibration |
| 9 | Explainability | **Partial:** logistic coefficients, XGBoost feature importance and native Tree SHAP, EBM global terms | Case-level review and EBM local contributions |
| 10 | Validation and backtesting | **Partial:** fixed calendar train/validation/test windows, separate late stress period, sampled holdout; no full-data backtest | Validate time windows, full-data evaluation, calibration, and repeat backtests |
| 11 | Monitoring and MLOps | **Partial:** reproducible run manifest, daily replay alert/label summaries, visible shift warning, and GitHub Actions CI | MLflow tracking, model versioning, and formal drift thresholds |
| 12 | Simulated real-time monitoring | **Partial:** chronological replay of saved scores with validation-only alert cutoff | Live or batch scoring of newly arriving transactions |
| 13 | Investigation dashboard and case management | **Partial:** local Streamlit model comparison, ranked alerts, transaction context, prior sender history and counterparty graph, explanations, and saved synthetic case notes | Review workflow polish and richer graph analysis |
| 14 | Investigation narratives and governance | **Partial:** sampled-study report, [model card](MODEL_CARD.md), assumptions, limitations, and error exports | Completed synthetic case narratives and formal validation report |

## Highest-priority remaining work

Fit and evaluate on the full primary period, repeat time windows, and estimate uncertainty with account dependence. Investigate why the [data profile](data_profile.md) changes sharply after September 10 before drawing conclusions from the late stress period. The [study report](STUDY_RESULTS.md) states what the current sample supports.
