import numpy as np
import pandas as pd
import pytest

from aml_monitoring.data import IBM_COLUMNS, load_ibm_csv, normalize_transactions
from aml_monitoring.evaluation import chronological_split, top_k_metrics
from aml_monitoring.features import build_features


def _transaction(timestamp, amount, receiver="R1", label=0, currency="USD"):
    return [timestamp, "001", "S1", "002", receiver, amount, "USD", amount,
            currency, "Wire", label]


def test_history_uses_strictly_earlier_timestamp_and_expires():
    frame = normalize_transactions(pd.DataFrame([
        _transaction("2024-01-01 00:00", 100),
        _transaction("2024-01-01 00:00", 200, "R2"),
        _transaction("2024-01-01 12:00", 300),
        _transaction("2024-01-09 00:00", 400),
    ], columns=IBM_COLUMNS))
    features = build_features(frame)
    assert features.loc[0, "prior_txn_count_24h"] == 0
    assert features.loc[1, "prior_txn_count_24h"] == 0
    assert features.loc[2, "prior_txn_count_24h"] == 2
    assert features.loc[2, "prior_amount_24h"] == pytest.approx(300)
    assert features.loc[2, "prior_avg_amount_7d"] == pytest.approx(150)
    assert features.loc[2, "new_counterparty_7d"] == 0
    assert features.loc[3, "prior_txn_count_7d"] == 0


def test_amount_history_only_uses_matching_payment_currency():
    frame = normalize_transactions(pd.DataFrame([
        _transaction("2024-01-01 00:00", 100, currency="USD"),
        _transaction("2024-01-01 01:00", 1000, currency="EUR"),
        _transaction("2024-01-01 02:00", 200, currency="USD"),
    ], columns=IBM_COLUMNS))
    features = build_features(frame)
    assert features.loc[2, "prior_txn_count_24h"] == 2
    assert features.loc[2, "prior_amount_24h"] == pytest.approx(100)
    assert features.loc[2, "prior_avg_amount_7d"] == pytest.approx(100)
    assert features.loc[2, "amount_vs_prior_avg_7d"] == pytest.approx(2)


def test_csv_loader_keeps_distinct_account_columns(tmp_path):
    path = tmp_path / "HI-Small_Trans.csv"
    path.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2024/01/01 00:00,001,ABC,002,XYZ,100,USD,100,USD,Wire,1\n"
    )
    row = load_ibm_csv(path).iloc[0]
    assert row["from_bank"] == "001"
    assert row["from_account"] == "ABC"
    assert row["to_account"] == "XYZ"
    assert row["is_laundering"] == 1


def test_csv_loader_rejects_changed_column_order(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Amount Paid,Receiving Currency,Amount Received,Payment Currency,Payment Format,Is Laundering\n"
        "2024/01/01 00:00,001,ABC,002,XYZ,100,USD,100,USD,Wire,1\n"
    )
    with pytest.raises(ValueError, match="11-column IBM"):
        load_ibm_csv(path)


def test_chronological_split_does_not_separate_equal_timestamps():
    times = pd.Series(pd.to_datetime([
        "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02",
        "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04",
        "2024-01-05", "2024-01-05",
    ]))
    split = chronological_split(times)
    assert np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(10))
    assert max(times.iloc[split["train"]]) < min(times.iloc[split["validation"]])
    assert max(times.iloc[split["validation"]]) < min(times.iloc[split["test"]])


def test_alert_capacity_counts_top_ranked_labels():
    result = top_k_metrics(np.array([0, 1, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]), 2)
    assert result["positives_found"] == 1
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 0.5
