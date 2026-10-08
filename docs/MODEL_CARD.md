# Model card: sampled IBM HI-Small XGBoost pilot

**Version:** 0.1 research pilot, 2026-10-08. **Selected specification:** XGBoost with transaction and prior-behavior features. This card describes the validation-selected model in the [study report](STUDY_RESULTS.md); no fitted model binary is distributed.

## Intended use

Rank **synthetic** transactions for a fixed-size educational review queue and compare ML baselines for a class project. A score indicates relative model priority within this dataset. It is **not** a calibrated probability, a suspicious activity report, or a decision about a real person or account. The model is not approved for real AML operations.

## Data and training

The source is IBM `HI-Small_Trans.csv`, profiled in [data_profile.md](data_profile.md). The main study randomly selects 10% of September 1–10 transaction rows (seed 42) for fitting and evaluation. DuckDB computes 24-hour and 7-day history using **all** earlier rows in the original 5,078,345-row CSV. Training uses September 1–6 (324,521 sampled rows, 260 positive labels); September 7–8 is validation; September 9–10 is the primary test. September 11–18 is a separate stress period because its volume and label prevalence shift sharply.

Inputs include log amount paid, hour, day of week, same-bank indicator, payment format and currencies, prior sender counts, same-payment-currency amount history, amount relative to that history, and a recent-new-counterparty indicator. Bank and account IDs build past activity but are not model inputs. Same-timestamp transactions are excluded from one another's history. The model has 200 trees, maximum depth 4, learning rate 0.05, and a positive-class weight derived from the training label ratio. See [models.py](../src/aml_monitoring/models.py) for the exact configuration.

## Evaluation

The specification was selected by **validation average precision** among currency-aware rules and three ML algorithms with transaction-only and behavioral feature sets. The primary holdout contains 86,066 sampled rows and 103 positive labels. Average precision is **0.2245**; ROC-AUC is **0.9733**. At 100 alerts, 23 labels are found (23% precision; 22.3% recall). A descriptive 300-repeat stratified row bootstrap gave a 95% AP interval of **0.1468–0.3128**. The interval does not capture account or time dependence. Exact metrics and segment diagnostics are generated under ignored `results/study_full_history_10pct/`.

The late stress period has 655 positive labels among 1,108 rows (59.1%), versus 0.1197% in the primary test. Its ranking metrics cannot be compared directly with the primary test. A saved-score replay uses a validation-only cutoff to study alert volume, but there is no probability calibration or prospective operational validation.

## Explanations

The study exports XGBoost feature importance and native Tree SHAP contributions for five top-scored primary-test transactions. Payment format, recent counterparty status, same-bank status, and sender history appear among influential inputs. These values explain the fitted model's score calculations in this synthetic dataset; they do not establish causal or investigative evidence. Logistic coefficients and EBM global terms are exported for the comparison models.

## Limits and next validation

- The modeling rows are sampled, even though earlier behavioral history is complete. Full-primary-period model evaluation remains pending.
- The data and labels are synthetic; real transaction distributions, alert workflows, and costs are untested.
- Positive labels are rare in the primary window; subgroup metrics with only a few positives are unstable.
- The dataset's final eight days have an unexplained volume and prevalence shift.
- Class weighting makes raw scores unsuitable as probabilities; calibration needs a separate validation design.
- Repeat across seeds and time windows, quantify account-aware uncertainty, review errors by segment, and document full-data memory and compute cost before making broader claims.

## Reproduction and governance

Use the [README](../README.md) to obtain the source CSV and run the study. The raw dataset, account-level predictions, local case notes, and fitted binaries are excluded from Git. The [local dashboard](../app/dashboard.py) can review generated synthetic alerts and save notes under ignored `results/`; no external report is filed or submitted.
