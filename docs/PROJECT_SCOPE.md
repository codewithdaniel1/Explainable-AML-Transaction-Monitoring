# Project scope and build status

Status as of 2026-10-08. This is the complete project plan discussed for the ML class and the larger portfolio system. The [walkthrough notebook](../notebooks/aml_pipeline_walkthrough.ipynb) teaches the core pipeline on **artificial demo data** and includes saved real-data outputs. The separate [IBM study](STUDY_RESULTS.md) is a **10% sampled pilot**. Raw CSVs and generated JSON and predictions are local artifacts excluded from Git; a fresh clone must follow the [README](../README.md).

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
| 14 | Investigation narratives and governance | **Partial:** sampled-study report, [model card](MODEL_CARD.md), [case template](CASE_REVIEW_TEMPLATE.md), assumptions, limitations, and error exports | Completed synthetic case narratives and formal validation report |

## Current course-project milestone

The [sampled pilot](STUDY_RESULTS.md) now compares those baselines and feature sets using fixed out-of-time windows, currency-consistent amounts, **complete earlier history** from DuckDB, validation-only model selection, and initial explanation and error analysis. It is a meaningful class-study milestone. It cannot support a full-data claim because model fitting and the primary holdout use sampled rows, and the late-period shift remains unexplained. The notebook remains a teaching artifact alongside the pilot.

## Why the full-data experiment is still pending

The [data profile](data_profile.md) found that September 11–18 contains only 1,108 transactions, of which 655 are labeled laundering. The pilot excludes this period from its primary holdout, but the cause of the shift remains unclear. Rolling amounts are currency-consistent and history is computed from all earlier rows on disk. A full-dataset fit and evaluation, repeated time windows, and account-aware uncertainty remain.
