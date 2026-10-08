"""Analyze held-out errors and uncertainty from a completed sampled study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import IBM_COLUMNS


def _load_test_predictions(path: Path, models: set[str]) -> pd.DataFrame:
    selected = []
    for chunk in pd.read_csv(path, chunksize=100_000):
        subset = chunk.loc[(chunk["period"] == "test") & chunk["model"].isin(models)]
        if not subset.empty:
            selected.append(subset)
    return pd.concat(selected, ignore_index=True)


def bootstrap_ap(y: np.ndarray, scores: np.ndarray, seed: int = 42,
                 repeats: int = 200) -> tuple[float, float]:
    """Stratified row bootstrap, keeping observed test prevalence fixed."""
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if not len(positives) or not len(negatives):
        raise ValueError("Bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        draw = np.concatenate((rng.choice(positives, len(positives)),
                               rng.choice(negatives, len(negatives))))
        values.append(average_precision_score(y[draw], scores[draw]))
    return tuple(float(v) for v in np.quantile(values, [.025, .975]))


def _extract_case_context(csv: Path, source_rows: set[int], output: Path) -> None:
    """Join a small case list to original synthetic transaction fields."""
    found = []
    offset = 0
    for chunk in pd.read_csv(csv, header=0, names=IBM_COLUMNS,
                             dtype=str, chunksize=100_000):
        positions = np.arange(offset, offset + len(chunk))
        mask = np.isin(positions, list(source_rows))
        if mask.any():
            part = chunk.loc[mask].copy()
            part.insert(0, "source_row", positions[mask])
            found.append(part)
        offset += len(chunk)
    if sum(len(part) for part in found) != len(source_rows):
        raise ValueError("Some case source rows were not found in the original CSV")
    pd.concat(found, ignore_index=True).to_csv(output, index=False)


def analyze_study(directory: Path, repeats: int = 200) -> dict:
    manifest = json.loads((directory / "study_manifest.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    selected = manifest["selected_by_validation_average_precision"]
    predictions = _load_test_predictions(directory / "predictions.csv", {selected})
    predictions = predictions.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    y = predictions["label"].to_numpy(dtype=np.int8)
    scores = predictions["score"].to_numpy()
    lower, upper = bootstrap_ap(y, scores, seed=manifest["seed"], repeats=repeats)
    k = min(100, len(predictions))
    top = predictions.head(k).copy()
    top["alert_rank"] = np.arange(1, k + 1)
    top.to_csv(directory / "top_100_alerts.csv", index=False)
    missed = predictions.iloc[k:].loc[lambda part: part["label"] == 1]
    near_misses = missed.head(100)
    near_misses.to_csv(directory / "highest_scored_missed_positives.csv", index=False)
    source = Path(manifest["source"])
    if source.exists():
        case_rows = set(pd.concat([top["source_row"], near_misses["source_row"]]).astype(int))
        _extract_case_context(source, case_rows, directory / "case_context.csv")
    segments = []
    for currency, group in predictions.groupby("payment_currency"):
        label_count = int(group["label"].sum())
        segments.append({
            "payment_currency": currency, "rows": len(group), "positives": label_count,
            "positive_rate": label_count / len(group),
            "average_precision": (float(average_precision_score(group["label"], group["score"]))
                                  if label_count and label_count < len(group) else None),
        })
    segments.sort(key=lambda row: -row["rows"])
    selected_metrics = metrics[selected]["test"]
    result = {
        "selected_model": selected,
        "primary_test_average_precision": selected_metrics["pr_auc"],
        "stratified_bootstrap_95pct_interval": [lower, upper],
        "bootstrap_repeats": repeats,
        "alert_capacity": k,
        "true_positives_at_capacity": int(top["label"].sum()),
        "false_positives_at_capacity": int(k - top["label"].sum()),
        "missed_positives_below_capacity": int(missed.shape[0]),
        "test_currency_segments": segments,
    }
    (directory / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Held-out diagnostics for the sampled pilot", "",
        f"Validation-selected model: **{selected}**. The primary test average precision is "
        f"**{selected_metrics['pr_auc']:.4f}**. A {repeats}-repeat stratified row bootstrap gives "
        f"a descriptive 95% interval of **{lower:.4f}–{upper:.4f}**. It does not account for "
        "account dependence, time dependence, or the loss of history caused by sampling.", "",
        f"At a capacity of {k} alerts: **{result['true_positives_at_capacity']}** labeled positives "
        f"and **{result['false_positives_at_capacity']}** labeled negatives were selected; "
        f"**{result['missed_positives_below_capacity']}** labeled positives remained below the cutoff.", "",
        "The generated `top_100_alerts.csv` and `highest_scored_missed_positives.csv` support "
        "case review. `case_context.csv` joins those identifiers to the original synthetic "
        "transaction fields when the raw CSV is available. These generated files are ignored by Git.", "",
        "## Payment-currency segments", "",
        "Small positive counts make individual segment AP estimates unstable; use these as error-analysis leads.", "",
        "| Payment currency | Rows | Positives | Positive rate | AP |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in segments:
        ap = "—" if row["average_precision"] is None else f"{row['average_precision']:.4f}"
        lines.append(f"| {row['payment_currency']} | {row['rows']:,} | {row['positives']} | "
                     f"{row['positive_rate']:.3%} | {ap} |")
    (directory / "diagnostics.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("results/study"))
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    result = analyze_study(args.study_dir, args.repeats)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
