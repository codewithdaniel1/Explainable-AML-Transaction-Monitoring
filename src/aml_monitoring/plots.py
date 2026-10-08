"""Publication-ready figures for the IBM data profile and sampled pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_daily_profile(profile_path: Path, output: Path) -> None:
    profile = json.loads(profile_path.read_text())
    days = pd.DataFrame.from_dict(profile["daily_activity"], orient="index")
    days.index = pd.to_datetime(days.index)
    rates = days["positives"] / days["rows"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    axes[0].bar(days.index, days["rows"], color="#27659d")
    axes[0].set_ylabel("Transactions")
    axes[0].set_title("IBM HI-Small daily volume and synthetic label rate")
    axes[1].plot(days.index, rates * 100, marker="o", color="#a83e34")
    axes[1].set_ylabel("Labeled laundering (%)")
    axes[1].set_xlabel("September 2022")
    axes[1].set_xticks(days.index[::2])
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].axvline(pd.Timestamp("2022-09-11"), color="#555", linestyle="--", linewidth=1)
    axes[1].grid(axis="y", alpha=.25)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_study(study_dir: Path, output: Path) -> None:
    manifest = json.loads((study_dir / "study_manifest.json").read_text())
    selected = manifest["selected_by_validation_average_precision"]
    comparisons = ["rules_behavioral", "xgboost_transaction", selected]
    comparisons = list(dict.fromkeys(comparisons))
    frames = []
    for chunk in pd.read_csv(study_dir / "predictions.csv", chunksize=100_000):
        part = chunk.loc[(chunk["period"] == "test") & chunk["model"].isin(comparisons)]
        if not part.empty:
            frames.append(part)
    predictions = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = ["#6c757d", "#4389bf", "#b7483e"]
    for color, model in zip(colors, comparisons):
        group = predictions.loc[predictions["model"] == model]
        y = group["label"].to_numpy()
        score = group["score"].to_numpy()
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        axes[0].step(recall, precision, where="post", color=color,
                     label=f"{model} (AP {ap:.3f})")
        ranked = y[np.argsort(-score, kind="stable")]
        ranks = np.arange(1, min(1000, len(ranked)) + 1)
        found = np.cumsum(ranked)[:len(ranks)]
        axes[1].plot(ranks, found, color=color, label=model)
    prevalence = predictions.loc[predictions["model"] == selected, "label"].mean()
    axes[0].axhline(prevalence, color="black", linestyle="--", linewidth=1,
                    label=f"Test prevalence ({prevalence:.4f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Primary test precision–recall")
    axes[0].set_yscale("log")
    axes[0].set_ylim(max(prevalence / 2, 1e-5), 1)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].axvline(100, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Alerts reviewed (K)", ylabel="Labeled positives found",
                title="Detections at review capacity")
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=.2)
    history_text = ("complete prior history" if manifest["full_history"]
                    else "sampled history is incomplete")
    fig.suptitle(f"{manifest['sample_fraction_primary']:.0%} sampled pilot; {history_text}", fontsize=11)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("results/ibm/data_profile.json"))
    parser.add_argument("--study-dir", type=Path,
                        default=Path("results/study_full_history_10pct"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_daily_profile(args.profile, args.output_dir / "ibm_daily_profile.png")
    plot_study(args.study_dir, args.output_dir / "sampled_pilot_performance.png")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
