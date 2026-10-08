# Class-project MVP scope

The original proposal explored a broader AML platform. For the ML class, the deliverable is a focused comparison of **logistic regression, XGBoost with Tree SHAP, and Explainable Boosting Machine (EBM)** on IBM's synthetic HI-Small transactions. Currency-aware rules provide a simple comparison. The main question is how detection and interpretability compare at a fixed alert-review capacity.

## In the MVP

1. [Profile the IBM data](data_profile.md), including class imbalance and the unusual final eight days.
2. Build transaction and strictly earlier account-behavior features. Use fixed chronological training, validation, and primary test periods.
3. Compare rules and the three ML models using average precision and Precision/Recall@100. Select a specification using validation data only.
4. Show the comparison, ranked examples, and explanations in the [walkthrough notebook](../notebooks/aml_pipeline_walkthrough.ipynb) and [study report](STUDY_RESULTS.md).
5. State the sampling, synthetic-data, and late-period limits clearly in the [model card](MODEL_CARD.md).

The present real-data result is a **10% sampled pilot**. Its behavioral features use complete earlier history, but model training and evaluation cover sampled rows. The [README](../README.md) has the commands to reproduce it. The artificial demo in the notebook teaches the code; it is not research evidence.

## Before calling the class study complete

Run the comparison on all eligible primary-period rows if compute permits, repeat it across seeds or time windows, investigate the late-period shift, and write the final class report. If full-data fitting is infeasible, report the sampled design and its limits explicitly.

## Optional later work

The existing saved-score replay and local dashboard are optional demonstrations. Databricks or Snowflake, dbt, MLflow, deep learning, graph models, live streaming, and case-management workflows belong to a later portfolio project. They are not requirements for this MVP.
