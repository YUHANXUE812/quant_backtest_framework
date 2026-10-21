"""Events retained as stable boundaries for a future event-driven engine."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    date: pd.Timestamp
    returns: pd.Series


@dataclass(frozen=True)
class SignalEvent:
    date: pd.Timestamp
    target_weights: pd.Series


@dataclass(frozen=True)
class OrderEvent:
    date: pd.Timestamp
    symbol: str
    weight_change: float


@dataclass(frozen=True)
class FillEvent:
    date: pd.Timestamp
    symbol: str
    filled_weight_change: float
    cost_rate: float

