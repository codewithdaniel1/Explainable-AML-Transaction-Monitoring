"""Comparable initial model specifications and a fixed rule baseline."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def _encoded_features(scale: bool, feature_columns: list[str]) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("numeric", numeric, [name for name in NUMERIC_FEATURES if name in feature_columns]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         [name for name in CATEGORICAL_FEATURES if name in feature_columns]),
    ])


def fit_model(name: str, x_train, y_train, seed: int = 42,
              feature_columns: list[str] | None = None):
    """Fit on the training period only; return an estimator with predict_proba."""
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training period must contain both classes")
    feature_columns = feature_columns or FEATURES
    if set(feature_columns) != set(x_train.columns):
        raise ValueError("Training columns must match feature_columns")
    if name == "logistic":
        model = Pipeline([
            ("prepare", _encoded_features(scale=True, feature_columns=feature_columns)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ])
        return model.fit(x_train, y_train)
    if name == "xgboost":
        from xgboost import XGBClassifier

        positives = int(np.sum(y_train))
        negatives = len(y_train) - positives
        model = Pipeline([
            ("prepare", _encoded_features(scale=False, feature_columns=feature_columns)),
            ("model", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=negatives / positives,
                tree_method="hist", n_jobs=4, random_state=seed,
            )),
        ])
        return model.fit(x_train, y_train)
    if name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier

        # EBM reads the named numeric and nominal columns directly, preserving
        # its additive terms for later explanation analysis.
        model = ExplainableBoostingClassifier(
            interactions=0, outer_bags=2, max_rounds=1500,
            n_jobs=4, random_state=seed,
        )
        positives = int(np.sum(y_train))
        negatives = len(y_train) - positives
        weights = np.where(np.asarray(y_train) == 1, len(y_train) / (2 * positives),
                           len(y_train) / (2 * negatives))
        return model.fit(x_train, y_train, sample_weight=weights)
    raise ValueError(f"Unknown model: {name}")


def rule_scores(features) -> np.ndarray:
    """Fixed, illustrative rules; no labels or holdout data set thresholds."""
    score = (
        (features["log_amount_paid"] >= np.log1p(10_000)).astype(int)
        + (features["prior_txn_count_24h"] >= 3).astype(int)
        + (features["amount_vs_prior_avg_7d"] >= 5).astype(int)
    )
    return (score / 3).to_numpy(dtype=float)
