# Project scope and build status

This is the complete project plan we discussed for the ML class and the larger portfolio system. The [walkthrough notebook](../notebooks/aml_pipeline_walkthrough.ipynb) runs the **working core pipeline** on artificial demo data and displays the **real IBM dataset profile**. It does not claim that every section below has been implemented.

| Section | Component | Current status | Next deliverable |
| ---: | --- | --- | --- |
| 1 | Dataset and exploratory analysis | **Partial:** IBM HI-Small CSV downloaded; counts, labels, time range, currencies, formats, and missingness profiled | Distributions, visualizations, and investigation of the late-period label shift |
| 2 | Databricks and dbt data engineering | **Planned** | Ingestion, bronze/silver/gold tables, dbt transformations and tests |
| 3 | Behavioral AML features | **Partial:** prior 24-hour and 7-day sender history, historical average, and new counterparty indicator | Currency-consistent amounts, more behavioral features, and feature-set comparison |
| 4 | Rule-based monitoring | **Partial:** fixed illustrative rules and alert-capacity evaluation | Currency-aware rules and validation-period threshold design |
| 5 | Supervised ML | **Partial:** logistic regression, XGBoost, and EBM run on the artificial demo | Train and tune on the full IBM data after resolving validation design |
| 6 | Deep learning | **Planned** | Separate MLP benchmark |
| 7 | Graph-based detection | **Planned** | Time-safe account graph features; possible GNN benchmark |
| 8 | Model performance metrics | **Partial:** PR-AUC, ROC-AUC, Brier score, Precision@K, Recall@K | Confusion matrices, error rates, computational cost, and uncertainty intervals |
| 9 | Explainability | **Planned** | Logistic coefficients, XGBoost SHAP, EBM global and local contributions |
| 10 | Validation and backtesting | **Partial:** chronological split and point-in-time feature checks | Investigate dataset tail, choose final holdout, calibration, segment analysis, and backtests if the time range supports them |
| 11 | Monitoring and MLOps | **Planned** | MLflow tracking, drift checks, model versioning, and CI |
| 12 | Simulated real-time monitoring | **Planned** | Replay transactions through scoring and alert generation |
| 13 | Investigation dashboard and case management | **Planned** | Streamlit demo with alerts, account history, notes, and graph views |
| 14 | Investigation narratives and governance | **Planned** | Synthetic draft notes, model card, assumptions, limitations, and validation report |

## Current course-project milestone

The first defensible research result will compare the rule baseline, logistic regression, XGBoost, and EBM on the IBM data using transaction-only and behavioral feature sets. It requires a documented out-of-time test design, currency-consistent amounts, validation-only tuning, and explanation and error analysis. The current notebook is a **teaching and pipeline-check artifact**, not that final experiment.

## Why the full-data experiment is still pending

The [data profile](data_profile.md) found that the final eight days have only 1,108 transactions, of which 655 are labeled laundering. This extreme shift can distort an ordinary chronological holdout. We need to investigate it and decide on a test window before reporting model comparisons.
