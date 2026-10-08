"""Replay saved scores chronologically using a validation-only alert threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def threshold_for_daily_capacity(validation: pd.DataFrame,
                                 daily_capacity: int) -> float:
    """Set the score cutoff from validation days without reading labels."""
    if daily_capacity <= 0 or validation.empty:
        raise ValueError("Need validation scores and a positive daily capacity")
    days = validation["timestamp"].dt.normalize().nunique()
    allowed = min(len(validation), daily_capacity * days)
    scores = validation["score"].to_numpy(dtype=float)
    return float(np.partition(scores, -allowed)[-allowed])


def replay_study(study_dir: Path, daily_capacity: int = 100) -> dict:
    manifest = json.loads((study_dir / "study_manifest.json").read_text())
    model = manifest["selected_by_validation_average_precision"]
    frames = []
    for chunk in pd.read_csv(study_dir / "predictions.csv", chunksize=100_000,
                             parse_dates=["timestamp"]):
        part = chunk.loc[chunk["model"] == model]
        if not part.empty:
            frames.append(part)
    predictions = pd.concat(frames, ignore_index=True)
    validation = predictions.loc[predictions["period"] == "validation"]
    cutoff = threshold_for_daily_capacity(validation, daily_capacity)
    replay = predictions.loc[predictions["period"].isin(["test", "late_stress"])].copy()
    replay = replay.sort_values(["timestamp", "source_row"], kind="stable")
    replay["alert"] = replay["score"] >= cutoff
    alerts = replay.loc[replay["alert"]].copy()
    alerts.to_csv(study_dir / "replay_alerts.csv", index=False)
    replay["date"] = replay["timestamp"].dt.date.astype(str)
    daily = replay.groupby(["date", "period"], sort=True).agg(
        transactions=("score", "size"), labels=("label", "sum"),
        alerts=("alert", "sum"),
    ).reset_index()
    found = replay.loc[replay["alert"]].groupby("date")["label"].sum()
    daily["labels_in_alerts"] = daily["date"].map(found).fillna(0).astype(int)
    daily["label_rate"] = daily["labels"] / daily["transactions"]
    daily["alert_rate"] = daily["alerts"] / daily["transactions"]
    daily.to_csv(study_dir / "replay_daily_monitoring.csv", index=False)
    result = {
        "selected_model": model,
        "validation_days": int(validation["timestamp"].dt.normalize().nunique()),
        "target_alerts_per_day_on_validation": daily_capacity,
        "validation_score_cutoff": cutoff,
        "replay_periods": {
            name: {"rows": int(len(group)), "alerts": int(group["alert"].sum()),
                   "labels_in_alerts": int(group.loc[group["alert"], "label"].sum())}
            for name, group in replay.groupby("period")
        },
        "note": "Replay uses saved scores. It does not retrain or score live transactions.",
    }
    (study_dir / "replay_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("results/study_full_history_10pct"))
    parser.add_argument("--daily-capacity", type=int, default=100)
    args = parser.parse_args()
    result = replay_study(args.study_dir, args.daily_capacity)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
