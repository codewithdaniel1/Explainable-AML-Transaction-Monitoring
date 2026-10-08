# IBM HI-Small transaction data profile

Profile run: 2026-10-07. Source: [IBM's Kaggle distribution](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), file `HI-Small_Trans.csv`. The raw CSV is stored locally at `data/raw/HI-Small_Trans.csv` and ignored by Git. SHA-256: `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`.

Run:

```bash
.venv/bin/aml --csv data/raw/HI-Small_Trans.csv --profile-only --output results/ibm
```

The full machine-readable output is `results/ibm/data_profile.json`.

## Dataset summary

| Measure | Result |
| --- | ---: |
| Transactions | 5,078,345 |
| Laundering-labeled transactions | 5,177 (0.102%) |
| Time range | 2022-09-01 00:00 to 2022-09-18 16:18 |
| Missing values in the 11 source fields | 0 |
| Payment formats | 7 |
| Payment currencies | 15 |
| Median amount paid | 1,414.54 in the source currency |
| 99th percentile amount paid | 13,524,530.71 in the source currency |
| Maximum amount paid | 1,046,302,363,293.48 in the source currency |

The amount statistics pool different currencies, including Bitcoin, so they are descriptive data checks, not comparable monetary values. Behavioral amount features and the fixed 10,000-unit rule need currency-specific handling before substantive model interpretation.

## Chronological split

| Period | Rows | Share | Labeled laundering | Label rate | Dates |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 3,047,216 | 60.00% | 2,299 | 0.0754% | Sep 1 to Sep 6, 13:36 |
| Validation | 1,015,565 | 20.00% | 1,081 | 0.1064% | Sep 6, 13:37 to Sep 8, 16:12 |
| Test | 1,015,564 | 20.00% | 1,797 | 0.1769% | Sep 8, 16:13 to Sep 18 |

Rows sharing a timestamp remain in the same period. All periods contain positive labels. The test label rate is more than twice the train rate, so prevalence-sensitive metrics and calibration require careful interpretation.

## Temporal anomaly to investigate

From September 1 through 10, the data contains 5,077,237 transactions and 4,522 positive labels (0.0891%). From September 11 through 18, it contains only **1,108 transactions, of which 655 are positive (59.1%)**. This abrupt change may reflect how the synthetic dataset was generated or ended; the profile alone does not establish the cause.

Before reporting model performance, examine daily label rates and transaction types, then define a defensible primary out-of-time window. Keep the late tail as a separate stress test if appropriate. Document the chosen window before tuning models and preserve an untouched final holdout.

The raw file is synthetic. Its labels support a controlled ML comparison, not a claim of real-world AML deployment readiness.
