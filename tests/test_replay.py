import pandas as pd
import pytest

from aml_monitoring.replay import threshold_for_daily_capacity


def test_replay_cutoff_uses_scores_and_validation_days_only():
    validation = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01",
                                     "2024-01-02", "2024-01-02", "2024-01-02"]),
        "score": [0.1, 0.4, 0.8, 0.2, 0.5, 0.9],
        "label": [1, 1, 0, 0, 0, 0],
    })
    assert threshold_for_daily_capacity(validation, 1) == pytest.approx(0.8)
    validation["label"] = 1 - validation["label"]
    assert threshold_for_daily_capacity(validation, 1) == pytest.approx(0.8)
