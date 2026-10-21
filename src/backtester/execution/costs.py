"""Turnover-based trading-cost model."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TransactionCostModel:
    commission_bps: float = 0.0
    sell_stamp_duty_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if min(self.commission_bps, self.sell_stamp_duty_bps, self.slippage_bps) < 0:
            raise ValueError("Trading-cost assumptions cannot be negative")

    def estimate(self, current: pd.Series, target: pd.Series) -> float:
        changes = target.subtract(current, fill_value=0.0)
        two_sided = changes.abs().sum() * (self.commission_bps + self.slippage_bps)
        sell_only = (-changes.clip(upper=0.0)).sum() * self.sell_stamp_duty_bps
        return float((two_sided + sell_only) / 10_000.0)

