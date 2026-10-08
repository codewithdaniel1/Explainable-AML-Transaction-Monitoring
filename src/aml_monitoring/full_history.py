"""Calculate sampled-row features from every earlier IBM transaction with DuckDB."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .data import IBM_RAW_HEADER
from .features import FEATURES


def build_full_history_features(csv: Path, sampled: pd.DataFrame,
                                scratch_root: Path, memory_limit: str = "2GB") -> pd.DataFrame:
    """Use disk-backed windows before selecting sampled rows.

    The CSV reader's ordinality is the original zero-based source row. Window
    frames exclude all same-timestamp peers, matching ``build_features``.
    """
    import duckdb

    if pd.read_csv(csv, nrows=0).columns.tolist() != IBM_RAW_HEADER:
        raise ValueError("Expected the 11-column IBM AML transaction CSV")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="aml_history_", dir=scratch_root) as temporary:
        temporary = Path(temporary)
        connection = duckdb.connect(
            str(temporary / "transactions.duckdb"),
            config={"memory_limit": memory_limit, "threads": "2",
                    "temp_directory": str(temporary / "spill")},
        )
        try:
            connection.execute("""
                CREATE TABLE transactions AS
                SELECT
                    src.ordinality - 1 AS source_row,
                    CAST(src."Timestamp" AS TIMESTAMP) AS timestamp,
                    src."From Bank" AS from_bank,
                    src."Account" AS from_account,
                    src."To Bank" AS to_bank,
                    src."Account_1" AS to_account,
                    CAST(src."Amount Paid" AS DOUBLE) AS amount_paid,
                    src."Payment Currency" AS payment_currency,
                    src."Receiving Currency" AS receiving_currency,
                    src."Payment Format" AS payment_format,
                    src."From Bank" || ':' || src."Account" AS sender,
                    src."To Bank" || ':' || src."Account_1" AS receiver
                FROM read_csv(?, all_varchar=true) WITH ORDINALITY AS src
            """, [str(csv)])
            selected = sampled[["source_row"]].copy()
            selected["position"] = range(len(selected))
            connection.register("selected", selected)
            result = connection.execute("""
                WITH rolling AS (
                    SELECT source_row, timestamp, from_bank, to_bank, amount_paid,
                           payment_format, payment_currency, receiving_currency,
                           count(*) OVER day_sender AS prior_txn_count_24h,
                           count(*) OVER week_sender AS prior_txn_count_7d,
                           sum(amount_paid) OVER day_currency AS prior_amount_24h,
                           avg(amount_paid) OVER week_currency AS prior_avg_amount_7d,
                           count(*) OVER week_counterparty AS prior_counterparty_count_7d
                    FROM transactions
                    WINDOW
                        day_sender AS (PARTITION BY sender ORDER BY timestamp
                            RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND CURRENT ROW EXCLUDE GROUP),
                        week_sender AS (PARTITION BY sender ORDER BY timestamp
                            RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW EXCLUDE GROUP),
                        day_currency AS (PARTITION BY sender, payment_currency ORDER BY timestamp
                            RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND CURRENT ROW EXCLUDE GROUP),
                        week_currency AS (PARTITION BY sender, payment_currency ORDER BY timestamp
                            RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW EXCLUDE GROUP),
                        week_counterparty AS (PARTITION BY sender, receiver ORDER BY timestamp
                            RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW EXCLUDE GROUP)
                )
                SELECT
                    ln(1 + rolling.amount_paid) AS log_amount_paid,
                    extract(hour FROM rolling.timestamp) AS hour,
                    extract(isodow FROM rolling.timestamp) - 1 AS day_of_week,
                    CAST(rolling.from_bank = rolling.to_bank AS INTEGER) AS same_bank,
                    rolling.prior_txn_count_24h,
                    greatest(coalesce(rolling.prior_amount_24h, 0), 0) AS prior_amount_24h,
                    rolling.prior_txn_count_7d,
                    coalesce(rolling.prior_avg_amount_7d, 0) AS prior_avg_amount_7d,
                    CASE WHEN rolling.prior_avg_amount_7d > 0
                         THEN rolling.amount_paid / rolling.prior_avg_amount_7d
                         ELSE 0 END AS amount_vs_prior_avg_7d,
                    CAST(rolling.prior_counterparty_count_7d = 0 AS INTEGER) AS new_counterparty_7d,
                    rolling.payment_format, rolling.payment_currency, rolling.receiving_currency
                FROM rolling
                JOIN selected USING (source_row)
                ORDER BY selected.position
            """).fetchdf()
            if len(result) != len(sampled):
                raise ValueError("Full-history query did not return every sampled row")
            if not np.allclose(result["log_amount_paid"].to_numpy(),
                               np.log1p(sampled["amount_paid"].to_numpy()), rtol=1e-10):
                raise ValueError("Full-history features do not align with sampled source rows")
            if not np.array_equal(result["payment_currency"].to_numpy(),
                                  sampled["payment_currency"].to_numpy()):
                raise ValueError("Full-history currencies do not align with sampled source rows")
            return result[FEATURES]
        finally:
            connection.close()
