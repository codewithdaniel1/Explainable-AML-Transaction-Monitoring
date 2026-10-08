import numpy as np
import pandas as pd
import pytest

from aml_monitoring.data import IBM_COLUMNS
from aml_monitoring.study import (currency_rule_scores, fit_currency_rules,
                                  period_indexes, sample_ibm_csv)
from aml_monitoring.features import build_features
from aml_monitoring.full_history import build_full_history_features


def test_calendar_periods_do_not_use_late_tail_for_selection():
    times = pd.Series(pd.to_datetime([
        "2022-09-06", "2022-09-07", "2022-09-08", "2022-09-09",
        "2022-09-10", "2022-09-11",
    ]))
    periods = period_indexes(times)
    assert [periods[name].tolist() for name in
            ("train", "validation", "test", "late_stress")] == [
                [0], [1, 2], [3, 4], [5],
            ]


def test_sampler_retains_tail_and_source_rows(tmp_path):
    path = tmp_path / "HI-Small_Trans.csv"
    frame = pd.DataFrame([
        ["2022/09/01 00:00", "001", "S1", "002", "R1", 100, "USD", 100, "USD", "Wire", 0],
        ["2022/09/11 00:00", "001", "S2", "002", "R2", 200, "EUR", 200, "EUR", "Wire", 1],
        ["2022/09/12 00:00", "001", "S3", "002", "R3", 300, "USD", 300, "USD", "Wire", 1],
    ], columns=IBM_COLUMNS)
    frame.to_csv(path, index=False, header=[
        "Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received",
        "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
    ])
    sample = sample_ibm_csv(path, fraction=.00001, seed=42, chunksize=2)
    assert sample["source_row"].tolist() == [1, 2]
    assert sample.attrs["source_rows"] == 3


def test_currency_rule_thresholds_fit_on_training_only():
    train = pd.DataFrame({"payment_currency": ["USD", "USD", "EUR", "EUR"],
                          "amount_paid": [100, 200, 1000, 2000]})
    thresholds = fit_currency_rules(train)
    features = pd.DataFrame({"prior_txn_count_24h": [0, 0],
                             "amount_vs_prior_avg_7d": [0, 0]})
    transactions = pd.DataFrame({"payment_currency": ["USD", "EUR"],
                                 "amount_paid": [500, 500]})
    scores = currency_rule_scores(features, transactions, thresholds)
    assert scores == pytest.approx(np.array([1 / 3, 0]))


def test_duckdb_full_history_matches_point_in_time_python_features(tmp_path):
    pytest.importorskip("duckdb")
    path = tmp_path / "HI-Small_Trans.csv"
    frame = pd.DataFrame([
        ["2024/01/01 00:00", "001", "S1", "002", "R1", 100, "USD", 100, "USD", "Wire", 0],
        ["2024/01/01 00:00", "001", "S1", "002", "R2", 200, "EUR", 200, "EUR", "Wire", 0],
        ["2024/01/01 01:00", "001", "S1", "002", "R1", 300, "USD", 300, "USD", "Wire", 1],
        ["2024/01/02 01:00", "001", "S1", "002", "R1", 400, "USD", 400, "USD", "Wire", 0],
    ], columns=IBM_COLUMNS)
    frame.to_csv(path, index=False, header=[
        "Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received",
        "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
    ])
    complete = sample_ibm_csv(path, fraction=1, seed=42)
    expected = build_features(complete)
    actual = build_full_history_features(path, complete, tmp_path)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    sampled = complete.iloc[[2, 3]].reset_index(drop=True)
    actual_sampled = build_full_history_features(path, sampled, tmp_path)
    pd.testing.assert_frame_equal(actual_sampled, expected.iloc[[2, 3]].reset_index(drop=True),
                                  check_dtype=False)
