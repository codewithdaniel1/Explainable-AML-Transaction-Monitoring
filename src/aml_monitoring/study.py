"""Reproducible, memory-bounded pilot study on IBM HI-Small transactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import IBM_COLUMNS, IBM_RAW_HEADER, normalize_transactions
from .evaluation import score_predictions
from .features import FEATURES, build_features
from .models import fit_model


TRANSACTION_FEATURES = [
    "log_amount_paid", "hour", "day_of_week", "same_bank",
    "payment_format", "payment_currency", "receiving_currency",
]
TRAIN_END = pd.Timestamp("2022-09-07")
VALIDATION_END = pd.Timestamp("2022-09-09")
PRIMARY_END = pd.Timestamp("2022-09-11")


def sample_ibm_csv(path: Path, fraction: float, seed: int,
                   chunksize: int = 100_000) -> pd.DataFrame:
    """Uniformly sample the primary period and retain the entire late tail."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if pd.read_csv(path, nrows=0).columns.tolist() != IBM_RAW_HEADER:
        raise ValueError("Expected the 11-column IBM AML transaction CSV")
    rng = np.random.default_rng(seed)
    string_columns = (
        "from_bank", "from_account", "to_bank", "to_account",
        "receiving_currency", "payment_currency", "payment_format",
    )
    chunks = []
    offset = 0
    for chunk in pd.read_csv(
        path, header=0, names=IBM_COLUMNS, chunksize=chunksize,
        dtype={name: "string" for name in string_columns},
    ):
        times = pd.to_datetime(chunk["timestamp"], errors="raise")
        sampled = rng.random(len(chunk)) < fraction
        keep = ((times < PRIMARY_END) & sampled) | (times >= PRIMARY_END)
        if keep.any():
            part = chunk.loc[keep].copy()
            part["source_row"] = np.flatnonzero(keep) + offset
            chunks.append(part)
        offset += len(chunk)
    if not chunks:
        raise ValueError("No transactions selected")
    frame = normalize_transactions(pd.concat(chunks, ignore_index=True))
    frame.attrs["source_rows"] = offset
    return frame


def period_indexes(timestamps: pd.Series) -> dict[str, np.ndarray]:
    """Calendar windows fixed before fitting or comparing models."""
    return {
        "train": np.flatnonzero((timestamps < TRAIN_END).to_numpy()),
        "validation": np.flatnonzero(((timestamps >= TRAIN_END) &
                                        (timestamps < VALIDATION_END)).to_numpy()),
        "test": np.flatnonzero(((timestamps >= VALIDATION_END) &
                                  (timestamps < PRIMARY_END)).to_numpy()),
        "late_stress": np.flatnonzero((timestamps >= PRIMARY_END).to_numpy()),
    }


def fit_currency_rules(train: pd.DataFrame) -> dict[str, float]:
    """99th-percentile paid amount per currency, estimated on training only."""
    return train.groupby("payment_currency")["amount_paid"].quantile(.99).to_dict()


def currency_rule_scores(features: pd.DataFrame, transactions: pd.DataFrame,
                         thresholds: dict[str, float]) -> np.ndarray:
    cutoff = transactions["payment_currency"].map(thresholds).to_numpy(dtype=float)
    amount = transactions["amount_paid"].to_numpy(dtype=float)
    large = np.isfinite(cutoff) & (amount >= cutoff)
    frequent = features["prior_txn_count_24h"].to_numpy() >= 3
    relative = features["amount_vs_prior_avg_7d"].to_numpy() >= 5
    return (large.astype(float) + frequent + relative) / 3


def explain_model(name: str, model, columns: list[str],
                  features: pd.DataFrame, indexes: np.ndarray,
                  scores: np.ndarray, source_rows: np.ndarray) -> dict:
    """Export global terms and small local examples for the held-out period."""
    if name == "logistic":
        names = model["prepare"].get_feature_names_out()
        values = model["model"].coef_[0]
        order = np.argsort(-np.abs(values))[:20]
        return {"scale": "standardized numeric / one-hot category",
                "top_coefficients": [{"feature": str(names[i]), "coefficient": float(values[i])}
                                     for i in order]}
    if name == "xgboost":
        names = model["prepare"].get_feature_names_out()
        gain = model["model"].feature_importances_
        order = np.argsort(-gain)[:20]
        result = {"top_importances": [{"feature": str(names[i]), "importance": float(gain[i])}
                                      for i in order]}
        # XGBoost's native pred_contribs are exact Tree SHAP log-odds terms.
        import xgboost as xgb

        chosen = indexes[np.argsort(-scores)[:min(5, len(indexes))]]
        matrix = model["prepare"].transform(features.iloc[chosen][columns])
        contributions = model["model"].get_booster().predict(
            xgb.DMatrix(matrix), pred_contribs=True
        )
        result["local_tree_shap_log_odds"] = [
            {"source_row": int(source_rows[row]), "base_value": float(values[-1]),
             "top_terms": [{"feature": str(names[i]), "contribution": float(values[i])}
                           for i in np.argsort(-np.abs(values[:-1]))[:10]]}
            for row, values in zip(chosen, contributions)
        ]
        return result
    if name == "ebm":
        importances = model.term_importances()
        order = np.argsort(-importances)[:20]
        return {"top_terms": [{"term": str(model.term_names_[i]),
                               "importance": float(importances[i])} for i in order]}
    raise ValueError(name)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_study(csv: Path, output: Path, fraction: float = .05,
              seed: int = 42, alerts: int = 100,
              full_history: bool = False) -> dict:
    """Fit fixed baselines on a sampled cohort and select by validation AP."""
    if alerts <= 0:
        raise ValueError("alerts must be positive")
    output.mkdir(parents=True, exist_ok=True)
    print("Sampling IBM CSV in chunks...", flush=True)
    transactions = sample_ibm_csv(csv, fraction, seed)
    total_rows = transactions.attrs["source_rows"]
    periods = period_indexes(transactions["timestamp"])
    labels = transactions["is_laundering"].to_numpy()
    for period, indexes in periods.items():
        positives = int(labels[indexes].sum())
        print(f"{period}: {len(indexes):,} sampled rows; {positives} positives", flush=True)
        if not len(indexes) or not positives or positives == len(indexes):
            raise ValueError(f"{period} needs both classes; increase the sample fraction")
    if full_history:
        from .full_history import build_full_history_features

        print("Computing complete currency-consistent history in DuckDB...", flush=True)
        features = build_full_history_features(csv, transactions, output)
    else:
        print("Building sample-conditioned, currency-consistent history...", flush=True)
        features = build_features(transactions)
    thresholds = fit_currency_rules(transactions.iloc[periods["train"]])
    metrics: dict = {}
    explanations: dict = {}
    predictions = []
    specs = [("rules", "behavioral", FEATURES)] + [
        (name, feature_set, columns)
        for feature_set, columns in (("transaction", TRANSACTION_FEATURES),
                                     ("behavioral", FEATURES))
        for name in ("logistic", "xgboost", "ebm")
    ]
    for name, feature_set, columns in specs:
        key = f"{name}_{feature_set}"
        print(f"Fitting {key}...", flush=True)
        model = None if name == "rules" else fit_model(
            name, features.iloc[periods["train"]][columns],
            labels[periods["train"]], seed=seed, feature_columns=columns,
        )
        metrics[key] = {}
        test_scores = None
        for period in ("validation", "test", "late_stress"):
            indexes = periods[period]
            scores = (currency_rule_scores(features.iloc[indexes], transactions.iloc[indexes],
                                           thresholds)
                      if model is None else model.predict_proba(features.iloc[indexes][columns])[:, 1])
            metrics[key][period] = score_predictions(labels[indexes], scores, alerts)
            predictions.append(pd.DataFrame({
                "source_row": transactions.iloc[indexes]["source_row"].to_numpy(),
                "timestamp": transactions.iloc[indexes]["timestamp"].to_numpy(),
                "period": period, "model": key, "label": labels[indexes], "score": scores,
                "payment_currency": transactions.iloc[indexes]["payment_currency"].to_numpy(),
            }))
            if period == "test":
                test_scores = scores
        if model is not None:
            explanations[key] = explain_model(
                name, model, columns, features, periods["test"], test_scores,
                transactions["source_row"].to_numpy(),
            )
        print(f"  validation AP={metrics[key]['validation']['pr_auc']:.4f}; "
              f"primary test AP={metrics[key]['test']['pr_auc']:.4f}", flush=True)
    selected = max(metrics, key=lambda key: metrics[key]["validation"]["pr_auc"])
    daily = transactions.groupby(transactions["timestamp"].dt.date)["is_laundering"].agg(
        ["size", "sum"]
    )
    manifest = {
        "source": str(csv), "original_csv_rows": total_rows,
        "sample_fraction_primary": fraction, "seed": seed,
        "full_history": full_history,
        "history_note": ("History includes all earlier rows in the original CSV."
                         if full_history else
                         "History uses sampled transactions only, so behavioral features are incomplete."),
        "periods": {name: {"rows": len(idx), "positives": int(labels[idx].sum())}
                    for name, idx in periods.items()},
        "date_boundaries_exclusive": {"train_end": str(TRAIN_END.date()),
                                      "validation_end": str(VALIDATION_END.date()),
                                      "primary_end": str(PRIMARY_END.date())},
        "daily_sample": {str(day): {"rows": int(row["size"]), "positives": int(row["sum"])}
                         for day, row in daily.iterrows()},
        "rule_amount_99th_percentile_by_currency": thresholds,
        "selected_by_validation_average_precision": selected,
    }
    _write_json(output / "study_manifest.json", manifest)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "explanations.json", explanations)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    _write_report(output / "report.md", manifest, metrics)
    return {"manifest": manifest, "metrics": metrics}


def _write_report(path: Path, manifest: dict, metrics: dict) -> None:
    rows = [
        "# IBM HI-Small sampled pilot study", "",
        "This is a **sampled pilot**, not full-dataset model performance. The primary period is "
        "September 1–10, 2022; September 11–18 is a separate stress period with a sharp label shift. "
        + ("Behavioral features include every earlier transaction in the original CSV."
           if manifest["full_history"] else
           "The sample excludes most prior transactions, so behavioral history is incomplete."), "",
        f"Original CSV rows read: {manifest['original_csv_rows']:,}. Primary sampling fraction: "
        f"{manifest['sample_fraction_primary']:.1%}. Seed: {manifest['seed']}.", "",
        "| Period | Calendar dates | Sampled rows | Positive labels |",
        "| --- | --- | ---: | ---: |",
    ]
    dates = {"train": "Sep 1–6", "validation": "Sep 7–8",
             "test": "Sep 9–10", "late_stress": "Sep 11–18"}
    for name, info in manifest["periods"].items():
        rows.append(f"| {name} | {dates[name]} | {info['rows']:,} | {info['positives']:,} |")
    rows += ["", "## Model comparison", "",
             "Average precision (AP) summarizes the precision–recall curve. The JSON key is `pr_auc`. "
             "Alert capacity is fixed at 100 per period. Models and feature sets are selected using validation AP only.", "",
             "| Model and features | Validation AP | Primary test AP | Test Recall@100 | Late stress AP |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in metrics.items():
        rows.append(f"| {name} | {values['validation']['pr_auc']:.4f} | "
                    f"{values['test']['pr_auc']:.4f} | {values['test']['recall_at_k']:.4f} | "
                    f"{values['late_stress']['pr_auc']:.4f} |")
    selected = manifest["selected_by_validation_average_precision"]
    rows += ["", f"Validation-selected specification: **{selected}**. Its primary test AP is "
             f"**{metrics[selected]['test']['pr_auc']:.4f}**.", "",
             "## Interpretation and limits", "",
             "- The late period has a different prevalence and very low volume. Its metrics are a stress check, not a comparable holdout.",
             ("- Primary rows were uniformly sampled across days, while behavioral history was computed from every original transaction. The test still contains only a sample of the primary period."
              if manifest["full_history"] else
              "- Randomly sampling primary rows preserves coverage across days but removes most account history. Behavioral features require a full-history run before research claims."),
             "- The rules use training-only 99th-percentile thresholds by payment currency, plus prior count and relative amount. These are illustrative thresholds.",
             "- Weighted model scores are not calibrated probabilities. Brier scores in the JSON are diagnostic.",
             "- This synthetic dataset does not establish real-world AML effectiveness.", ""]
    path.write_text("\n".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/study"))
    parser.add_argument("--fraction", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alerts", type=int, default=100)
    parser.add_argument("--full-history", action="store_true",
                        help="Use disk-backed DuckDB windows over all original rows")
    args = parser.parse_args()
    run_study(args.csv, args.output, args.fraction, args.seed, args.alerts,
              full_history=args.full_history)


if __name__ == "__main__":
    main()
