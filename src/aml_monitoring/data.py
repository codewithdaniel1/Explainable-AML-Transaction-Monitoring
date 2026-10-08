"""IBM AML CSV ingestion and a small, clearly synthetic demo fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IBM_COLUMNS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid", "payment_currency",
    "payment_format", "is_laundering",
]
IBM_RAW_HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]


def load_ibm_csv(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load HI-Small_Trans.csv, preserving bank and account IDs as text.

    A row cap is for development only; results from a capped prefix are not a
    valid estimate of performance on the full chronological holdout.
    """
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != IBM_RAW_HEADER:
        raise ValueError("Expected the 11-column IBM AML transaction CSV")
    frame = pd.read_csv(
        path,
        header=0,
        names=IBM_COLUMNS,
        dtype={name: "string" for name in (
            "from_bank", "from_account", "to_bank", "to_account",
            "receiving_currency", "payment_currency", "payment_format",
        )},
        nrows=max_rows,
    )
    return normalize_transactions(frame)


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(IBM_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    for name in ("amount_paid", "amount_received"):
        out[name] = pd.to_numeric(out[name], errors="raise")
    out["is_laundering"] = pd.to_numeric(out["is_laundering"], errors="raise")
    if out.empty or out["timestamp"].isna().any():
        raise ValueError("Transactions must contain valid timestamps")
    if out["is_laundering"].isna().any() or not out["is_laundering"].isin([0, 1]).all():
        raise ValueError("is_laundering must contain only 0 and 1")
    if out[["amount_paid", "amount_received"]].isna().any().any():
        raise ValueError("Transaction amounts cannot be missing")
    if (out[["amount_paid", "amount_received"]] < 0).any().any():
        raise ValueError("Transaction amounts cannot be negative")
    for name in ("from_bank", "from_account", "to_bank", "to_account",
                 "receiving_currency", "payment_currency", "payment_format"):
        if out[name].isna().any():
            raise ValueError(f"{name} cannot be missing")
        out[name] = out[name].astype(str)
    out["is_laundering"] = out["is_laundering"].astype("int8")
    # A stable sort preserves CSV order for equal timestamps, while feature
    # construction treats every transaction at that timestamp as simultaneous.
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def make_demo_transactions(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate toy transactions for a smoke test, not an AML benchmark."""
    if n < 30:
        raise ValueError("Demo requires at least 30 transactions")
    rng = np.random.default_rng(seed)
    senders = rng.integers(0, 45, n)
    receivers = rng.integers(0, 55, n)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="30min")
    labels = (rng.random(n) < 0.07).astype("int8")
    # A learnable toy pattern lets us check that the pipeline executes; it is
    # deliberately artificial and must never be reported as research evidence.
    amounts = np.where(labels == 1, rng.lognormal(10, 0.6, n), rng.lognormal(7, 0.8, n))
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "from_bank": np.where(senders % 3 == 0, "001", "002"),
        "from_account": [f"S{x:04d}" for x in senders],
        "to_bank": np.where(receivers % 3 == 0, "001", "003"),
        "to_account": [f"R{x:04d}" for x in receivers],
        "amount_received": amounts,
        "receiving_currency": np.where(receivers % 4 == 0, "EUR", "USD"),
        "amount_paid": amounts,
        "payment_currency": "USD",
        "payment_format": np.where(labels == 1, "Wire", "ACH"),
        "is_laundering": labels,
    })
    return normalize_transactions(frame)
