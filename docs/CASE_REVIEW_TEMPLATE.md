# Synthetic alert review template

Use this template only for the IBM synthetic dataset. A model score starts a review; it is not evidence of wrongdoing or a real suspicious activity report. The [dashboard](../app/dashboard.py) saves short local notes under ignored `results/`.

## Alert identity and context

- Original CSV `source_row` and study run/seed:
- Review rank, capacity, and model specification:
- Transaction time, payment format, payment and receiving currencies:
- Amount paid and amount received, each with its own currency:
- Sender and receiver IDs (keep account-level details in ignored local files):

## Prior activity available at scoring time

- Sender transactions in the prior 24 hours and 7 days:
- Prior paid amount in the **same payment currency**:
- Amount relative to the same-currency 7-day average:
- Whether the receiver was new in the prior 7 days:
- Other linked synthetic transactions or patterns checked:

## Model information

- Score and its rank within the reviewed period:
- Leading local Tree SHAP contributions or EBM terms, if available:
- Plausible alternative explanations or missing context:

## Analyst assessment

- Disposition: unreviewed / needs further review / dismissed
- Factual observations from the synthetic record:
- Open questions and suggested next inspection:
- Reviewer and date:

Keep the synthetic label separate from the analyst assessment when drafting a blind review. Do not describe model importance as causal evidence. No automated external report or case filing is implemented.
