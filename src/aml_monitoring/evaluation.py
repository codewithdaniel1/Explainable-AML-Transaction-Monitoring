"""Chronological evaluation with a fixed alert-review capacity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def chronological_split(timestamps: pd.Series, train_fraction: float = 0.6,
                        validation_fraction: float = 0.2) -> dict[str, np.ndarray]:
    """Target row fractions while keeping equal-time rows together."""
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid split fractions")
    values = timestamps.to_numpy()
    unique_times, first_rows = np.unique(values, return_index=True)
    if len(unique_times) < 3:
        raise ValueError("Need at least three distinct timestamps")
    first = min(max(1, int(np.searchsorted(first_rows, len(values) * train_fraction))),
                len(unique_times) - 2)
    second = min(max(first + 1, int(np.searchsorted(
        first_rows, len(values) * (train_fraction + validation_fraction)))),
        len(unique_times) - 1)
    train_end = unique_times[first]
    validation_end = unique_times[second]
    return {
        "train": np.flatnonzero(values < train_end),
        "validation": np.flatnonzero((values >= train_end) & (values < validation_end)),
        "test": np.flatnonzero(values >= validation_end),
    }


def top_k_metrics(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    if len(y_true) != len(scores) or k <= 0:
        raise ValueError("Invalid labels, scores, or alert capacity")
    k = min(k, len(scores))
    top = np.argsort(-scores, kind="stable")[:k]
    found = int(np.asarray(y_true)[top].sum())
    positives = int(np.asarray(y_true).sum())
    return {
        "alerts": k,
        "positives_found": found,
        "precision_at_k": found / k,
        "recall_at_k": found / positives if positives else None,
    }


def score_predictions(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite values between 0 and 1")
    result = top_k_metrics(y_true, scores, k)
    result.update({
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, scores)) if y_true.sum() else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "brier_score": float(brier_score_loss(y_true, scores)),
    })
    return result
