# Explainable AML Transaction Monitoring: Class Report

## Research question

Can three simple detection tracks provide useful signals to understandable AML models? This project compares logistic regression, XGBoost, and an Explainable Boosting Machine (EBM), then asks whether alerts from the rules, anomaly, or network tracks find labeled positives that the selected model misses without creating too much extra review work. The [project guide's architecture diagram](PROJECT_GUIDE.md#architecture-at-a-glance) shows the complete flow.

## Stage 1: Data profile, features, and time split

The [SAML-D dataset](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D) contains **9,504,852 synthetic transactions**, with **9,873 positive labels (0.1039%)**, from October 7, 2022 through August 23, 2023. It includes payment type, amount, currencies, bank locations, account IDs, and a generated laundering typology. The notebook's EDA shows different positive-label rates across payment methods: 0.6239% for Cash Deposit and 0.0577% for ACH. Within UK-pound payments, the median positive-label amount is about 5,091, below the negative-label median of about 6,047. These observations describe the generator, not real bank risk.

Features include current payment facts and counts or averages from **strictly earlier** account activity. Earlier distinct counterparties provide simple fan-in and fan-out network features. Raw account IDs establish relationships but are not direct classifier inputs. `Laundering_type` is reserved for the final coverage audit and is never used to flag or train a transaction.

The notebook uses every row and splits complete days chronologically:

| Period | Dates | Transactions | Positive labels | Positive rate |
| --- | --- | ---: | ---: | ---: |
| Training | Oct 7, 2022–May 18, 2023 | 6,661,223 | 6,774 | 0.1017% |
| Validation | May 19–Jul 5, 2023 | 1,430,805 | 1,416 | 0.0990% |
| Testing | Jul 6–Aug 23, 2023 | 1,412,824 | 1,683 | 0.1191% |

## Stage 2: Three parallel detection tracks

The tracks create **signals for the models** and also produce their own yes/no alerts for a separate evaluation. Their alerts do not filter transactions before modeling: each classifier scores every validation and test row.

### Track 1: Fixed rules

Three readable flags identify cash deposits into accounts with many earlier distinct senders, changed cash-withdrawal amounts, and new cross-border currency-mismatched account pairs. Their thresholds are illustrative and use no outcome labels.

### Track 2: Isolation Forest anomaly score

Two Isolation Forests fit without labels on the first 10% of training days. The original broad `anomaly_score` remains a classifier feature. A second forest uses amount, change from the sender’s earlier usual amount, time, and same-location/currency behavior for standalone alerts. We selected that compact feature set using validation labels. Its alert cutoff is the 99.9th percentile of validation scores, computed without labels.

### Track 3: Account-network fan-in/out

Earlier distinct counterparties yield one-hop network features. The original largest-degree `network_score` remains a classifier feature. The standalone alert score divides the sum of log prior distinct counterparties by one plus the log prior sender–receiver pair count, so repeated activity within one pair is less prominent. We chose this formula using validation labels and set its cutoff at the unlabeled validation 99.9th percentile. This is simple relationship analysis, not multi-hop graph analytics.

The standalone alerts from all three tracks were evaluated on the same later test period. **Caught** means alerted positive synthetic labels; **false alerts** means alerted negative labels.

| Track | Alerts | Caught | False alerts | Precision | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed rules | 69,708 | 648 | 69,060 | 0.93% | 38.5% | 4.894% |
| Isolation Forest | 1,323 | 45 | 1,278 | 3.40% | 2.67% | 0.0906% |
| Network fan-in/out | 2,075 | 792 | 1,283 | 38.17% | 47.06% | 0.0909% |

Before this alert redesign, Isolation Forest caught 5 positives with 2,340 false alerts (0.21% precision, 0.30% recall, 0.1658% FPR); the network alert caught none with 2,112 false alerts. Both revised tracks improve precision, recall, and FPR on the later test. **The classifier features were held fixed**, so this is an improvement to the standalone alert tracks, not a new XGBoost model.

## Stage 3: Compare three risk-prioritization models

**The three tracks are not the three classifiers.** Logistic regression, XGBoost, and EBM each use the same base transaction features, individual rule flags, the original anomaly score, and the original network features. The alert-focused anomaly and network scores are not model inputs, so the revised track alerts leave all three classifier inputs and metrics unchanged. Three expanding training-time folds select the model family by mean average precision (AP). All three models then fit on the full training period. Each model's validation threshold maximizes F1; the later test period supplies final metrics.

XGBoost won the time-fold comparison with mean AP **0.9127**, versus **0.4636** for EBM and **0.1401** for logistic regression. On the later test period, each model used its own validation-selected threshold:

| Model | Test AP | Precision | Recall | F1 | FPR | Alerts | Caught | False alerts | Missed of 1,683 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.3176 | 44.2% | 29.8% | 0.3557 | 0.0449% | 1,134 | 501 | 633 | 1,182 |
| **XGBoost** | **0.9900** | **96.6%** | **98.0%** | **0.9732** | **0.0041%** | 1,708 | **1,650** | **58** | **33** |
| EBM | 0.7815 | 81.5% | 65.9% | 0.7286 | 0.0179% | 1,361 | 1,109 | 252 | 574 |

## Stage 4: Diagnostic alert comparison

**XGBoost is the selected model and produces 1,708 test alerts.** This section asks what would happen if the standalone track alerts were added to its queue; it does not propose doing so. The **three-track union** alerts when **fixed rules OR Isolation Forest OR network fan-in/out** alerts for a transaction. A transaction flagged by multiple tracks counts once. This union contains **only track alerts**, not XGBoost alerts.

| Combined track alerts | Alerts | Caught | False alerts | Precision | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Three-track union** | **72,328** | **1,196** | **71,132** | **1.65%** | **71.1%** | **5.041%** |

The three individual track rows total 73,106 alerts; removing 778 duplicate counts leaves **72,328 unique alerted transactions**. Of those, 1,196 have positive synthetic labels and 71,132 have negative labels. Precision is `1,196 / 72,328 = 1.65%`; recall is `1,196 / 1,683 = 71.1%`; FPR is `71,132 / 1,411,141 = 5.041%`.

We then compare track alerts with **XGBoost's own alerts**. Rules find **11 positive labels** that XGBoost missed but add **69,004 false alerts**. Isolation Forest and network fan-in/out each find **zero additional positives** beyond XGBoost, even though their standalone results improved substantially; their true-positive alerts overlap XGBoost’s. Combining all three tracks with XGBoost catches **1,661/1,683 positives (98.7% recall)** but expands the queue to **72,795 alerts** with only **2.28% precision**. This combined queue illustrates the coverage and workload tradeoff; it is not a recommended alert policy.

## Stage 5: Audit misses and explain model scores

XGBoost's **33 missed positive labels** include 10 Cash Deposits, 9 Cross-border transactions, and 4 each of ACH and Credit card. The largest generated-typology gaps are Smurfing (10 misses) and Layered_Fan_Out (8). `Laundering_type` was used only for this post-score audit.

The [notebook](../aml_modeling.ipynb) shows separate model scorecards and explanation charts: logistic regression coefficients and one-row contributions, global and local Tree SHAP for XGBoost, and InterpretML's native EBM global importance, feature effect, and local explanations. These show how each model formed its score; they do not establish whether laundering occurred.

## How to read the metrics

An **alert** is a transaction whose model score or track signal crosses its threshold. A **true positive** is an alerted positive *synthetic dataset label*; a **false positive** is an alerted negative label; a **missed positive** is a positive-label transaction without an alert. These terms do not establish criminal activity.

- **AP** summarizes ranking over many thresholds. The test label rate, 0.1191%, is an approximate random-ranking baseline.
- **Precision** is true positives divided by alerts. **Recall** is true positives divided by all 1,683 test positive labels.
- **F1** balances precision and recall at the validation-selected threshold.
- **False positive rate (FPR)** is false positives divided by the 1,411,141 negative-label test transactions. It should be read alongside the absolute alert workload.
- **Additional positives** from a track are labels XGBoost missed but the track flagged. **Additional false alerts** measure the corresponding extra workload.

## Limits and conclusion

XGBoost's 0.9900 test AP is exceptionally high for a rare-label task. The inputs contain neither the target nor the generated typology field, but SAML-D may encode clean patterns that make its labels easier to predict than real investigation outcomes. The alert redesign improved standalone track detection but did not change the classifiers’ feature values or their scores. Its improvement was selected on validation labels and confirmed on this one later test split; repeating the assessment on another time period would better establish stability. This experiment does **not** isolate which earlier feature caused the classifier improvement from older notebook versions; that would require a separate ablation. The result applies to this synthetic CSV and time split, not to a live bank.

The workflow is an educational approximation of detection and prioritization. It does not implement Actimize, Mantas, multi-hop graph analytics, an analyst console, case disposition, or a feedback loop. Real monitoring also needs customer context, investigation outcomes, risk-coverage review, capacity planning, independent validation, data quality controls, and ongoing monitoring. No universal U.S. AML rule prescribes a precision, recall, F1, or FPR target; the [FFIEC BSA/AML Examination Manual](https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04) describes risk-based monitoring.

**Dataset citation:** Oztas et al., “Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset,” *2023 IEEE International Conference on e-Business Engineering*, DOI [10.1109/ICEBE59045.2023.00028](https://doi.org/10.1109/ICEBE59045.2023.00028). The author notes that the distributed dataset is an updated version of the one used in the paper.
