"""Command-line entry point for profiling and initial model comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_ibm_csv, make_demo_transactions
from .evaluation import chronological_split, score_predictions
from .features import FEATURES, build_features
from .models import fit_model, rule_scores


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(args: argparse.Namespace) -> None:
    transactions = make_demo_transactions(args.demo_rows, args.seed) if args.demo else load_ibm_csv(
        args.csv, max_rows=args.max_rows
    )
    if args.max_rows is not None and not args.demo:
        print("DEVELOPMENT PREFIX: metrics from --max-rows are not research results")
    split = chronological_split(transactions["timestamp"])
    profile = {
        "source": "artificial_demo" if args.demo else str(args.csv),
        "development_prefix": bool(args.max_rows is not None and not args.demo),
        "rows": len(transactions),
        "positives": int(transactions["is_laundering"].sum()),
        "positive_rate": float(transactions["is_laundering"].mean()),
        "first_timestamp": str(transactions["timestamp"].min()),
        "last_timestamp": str(transactions["timestamp"].max()),
        "missing_values": {column: int(count) for column, count in
                           transactions.isna().sum().items()},
        "payment_formats": {str(key): int(value) for key, value in
                            transactions["payment_format"].value_counts().items()},
        "payment_currencies": {str(key): int(value) for key, value in
                               transactions["payment_currency"].value_counts().items()},
        "amount_paid_quantiles": {str(q): float(value) for q, value in
                                  transactions["amount_paid"].quantile([0, .5, .9, .99, 1]).items()},
        "splits": {},
    }
    daily = transactions.groupby(transactions["timestamp"].dt.date)["is_laundering"].agg(
        ["size", "sum"]
    )
    profile["daily_activity"] = {
        str(day): {"rows": int(row["size"]), "positives": int(row["sum"])}
        for day, row in daily.iterrows()
    }
    for name, indexes in split.items():
        profile["splits"][name] = {
            "rows": len(indexes),
            "positives": int(transactions.iloc[indexes]["is_laundering"].sum()),
            "positive_rate": float(transactions.iloc[indexes]["is_laundering"].mean()),
            "row_fraction": len(indexes) / len(transactions),
            "first_timestamp": str(transactions.iloc[indexes]["timestamp"].min()),
            "last_timestamp": str(transactions.iloc[indexes]["timestamp"].max()),
        }
    if args.profile_only:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "data_profile.json", profile)
        print(json.dumps(profile, indent=2))
        print(f"Wrote data profile to {output / 'data_profile.json'}")
        return

    print(f"Rows: {profile['rows']:,}; positive labels: {profile['positives']:,}")

    features = build_features(transactions)
    labels = transactions["is_laundering"].to_numpy()
    if any(profile["splits"][name]["positives"] == 0 for name in split):
        raise ValueError("A chronological period has no positive labels; inspect the data before training")
    train = split["train"]
    metrics = {}
    for name in args.models:
        model = None if name == "rules" else fit_model(name, features.iloc[train][FEATURES], labels[train], args.seed)
        metrics[name] = {}
        for period in ("validation", "test"):
            indexes = split[period]
            x = features.iloc[indexes][FEATURES]
            scores = rule_scores(x) if model is None else model.predict_proba(x)[:, 1]
            metrics[name][period] = score_predictions(labels[indexes], scores, args.alerts)
        print(f"{name}: test average precision={metrics[name]['test']['pr_auc']:.4f}; "
              f"Recall@{metrics[name]['test']['alerts']}="
              f"{metrics[name]['test']['recall_at_k']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Path to IBM HI-Small_Trans.csv")
    source.add_argument("--demo", action="store_true", help="Use artificial smoke-test transactions")
    parser.add_argument("--demo-rows", type=int, default=1200)
    parser.add_argument("--max-rows", type=int, default=None, help="Development-only CSV prefix")
    parser.add_argument("--output", default="results/ibm", help="Profile output directory with --profile-only")
    parser.add_argument("--alerts", type=int, default=100, help="Fixed alert capacity per evaluation period")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=["rules", "logistic", "xgboost", "ebm"],
                        default=["rules", "logistic", "xgboost", "ebm"])
    args = parser.parse_args()
    if args.demo and args.max_rows is not None:
        parser.error("--max-rows applies only to --csv")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    run(args)


if __name__ == "__main__":
    main()
