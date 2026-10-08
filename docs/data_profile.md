# IBM HI-Small transaction data profile

Profile run: 2026-10-07. Source: [IBM's Kaggle distribution](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), file `HI-Small_Trans.csv`. SHA-256: `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`.

The raw CSV and generated `results/ibm/data_profile.json` are ignored by Git. These tables preserve the main results; follow the [README](../README.md#download-and-profile-the-ibm-data) to regenerate the JSON.

## Dataset summary

| Measure | Result |
| --- | ---: |
| Transactions | 5,078,345 |
| Laundering-labeled transactions | 5,177 (0.102%) |
| Time range | 2022-09-01 00:00 to 2022-09-18 16:18 |
| Missing values in the 11 source fields | 0 |
| Payment formats | 7 |
| Payment currencies | 15 |
| Median amount paid | 1,414.54 raw units (currencies mixed) |
| 99th percentile amount paid | 13,524,530.71 raw units (currencies mixed) |
| Maximum amount paid | 1,046,302,363,293.48 raw units (currencies mixed) |

The amount statistics pool different currencies, including Bitcoin, so they are descriptive data checks, not comparable monetary values. The current feature builder now keeps rolling amount history within each payment currency. The original demo rule still has a fixed 10,000-unit cutoff; the [sampled IBM pilot](STUDY_RESULTS.md) uses training-only thresholds by currency.

## Chronological split

| Period | Rows | Share | Labeled laundering | Label rate | Dates |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 3,047,216 | 60.00% | 2,299 | 0.0754% | Sep 1 to Sep 6, 13:36 |
| Validation | 1,015,565 | 20.00% | 1,081 | 0.1064% | Sep 6, 13:37 to Sep 8, 16:12 |
| Test | 1,015,564 | 20.00% | 1,797 | 0.1769% | Sep 8, 16:13 to Sep 18 |

This is the introductory CLI's 60/20/20 **profile split**, which keeps equal timestamps together. It is **not** the calendar split used for the [sampled model study](STUDY_RESULTS.md). All periods contain positive labels. The test label rate is more than twice the train rate, so prevalence-sensitive metrics and calibration require care. These are counts, not model evaluation results.

## Temporal anomaly to investigate

From September 1 through 10, the data contains 5,077,237 transactions and 4,522 positive labels (0.0891%). From September 11 through 18, it contains only **1,108 transactions, of which 655 are positive (59.1%)**. This abrupt change may reflect how the synthetic dataset was generated or ended; the profile alone does not establish the cause.

The [sampled study](STUDY_RESULTS.md) uses September 1–10 for its primary comparison and keeps September 11–18 as a separate stress check. The cause of the shift still needs investigation before broader claims.

The raw file is synthetic. Its labels support a controlled ML comparison, not a claim of real-world AML deployment readiness.
