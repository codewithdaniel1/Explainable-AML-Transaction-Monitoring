"""Point-in-time features built before any train/validation/test split."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "log_amount_paid", "hour", "day_of_week", "same_bank",
    "prior_txn_count_24h", "prior_amount_24h", "prior_txn_count_7d",
    "prior_avg_amount_7d", "amount_vs_prior_avg_7d", "new_counterparty_7d",
]
CATEGORICAL_FEATURES = ["payment_format", "payment_currency", "receiving_currency"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class _SenderHistory:
    day: deque = field(default_factory=deque)
    week: deque = field(default_factory=deque)
    day_amount: float = 0.0
    week_amount: float = 0.0
    counterparties: Counter = field(default_factory=Counter)

    def expire(self, now: pd.Timestamp) -> None:
        day_start = now - pd.Timedelta(days=1)
        week_start = now - pd.Timedelta(days=7)
        while self.day and self.day[0][0] < day_start:
            _, amount = self.day.popleft()
            self.day_amount -= amount
        while self.week and self.week[0][0] < week_start:
            _, amount, receiver = self.week.popleft()
            self.week_amount -= amount
            self.counterparties[receiver] -= 1
            if self.counterparties[receiver] == 0:
                del self.counterparties[receiver]

    def add(self, now: pd.Timestamp, amount: float, receiver: str) -> None:
        self.day.append((now, amount))
        self.week.append((now, amount, receiver))
        self.day_amount += amount
        self.week_amount += amount
        self.counterparties[receiver] += 1


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return model features using strictly earlier timestamps as history.

    Transactions sharing a timestamp are scored against the same prior state,
    then added together. This avoids leaking an arbitrary CSV row order.
    """
    if not transactions["timestamp"].is_monotonic_increasing:
        raise ValueError("Transactions must be sorted by timestamp")
    histories: dict[str, _SenderHistory] = defaultdict(_SenderHistory)
    rows: list[dict] = []
    for now, batch in transactions.groupby("timestamp", sort=False):
        pending = []
        for txn in batch.itertuples(index=False):
            sender = f"{txn.from_bank}:{txn.from_account}"
            receiver = f"{txn.to_bank}:{txn.to_account}"
            history = histories[sender]
            history.expire(now)
            prior_average = history.week_amount / len(history.week) if history.week else 0.0
            amount = float(txn.amount_paid)
            rows.append({
                "log_amount_paid": np.log1p(amount),
                "hour": now.hour,
                "day_of_week": now.dayofweek,
                "same_bank": int(txn.from_bank == txn.to_bank),
                "prior_txn_count_24h": len(history.day),
                "prior_amount_24h": max(0.0, history.day_amount),
                "prior_txn_count_7d": len(history.week),
                "prior_avg_amount_7d": prior_average,
                "amount_vs_prior_avg_7d": amount / prior_average if prior_average > 0 else 0.0,
                "new_counterparty_7d": int(receiver not in history.counterparties),
                "payment_format": txn.payment_format,
                "payment_currency": txn.payment_currency,
                "receiving_currency": txn.receiving_currency,
            })
            pending.append((sender, amount, receiver))
        for sender, amount, receiver in pending:
            histories[sender].add(now, amount, receiver)
    return pd.DataFrame.from_records(rows, columns=FEATURES)
