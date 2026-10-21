"""Cross-sectional A-share momentum strategy."""

from dataclasses import dataclass

import pandas as pd

from backtester.portfolio.allocation import equal_weight_selection
from backtester.signals.transforms import quantile_groups
from backtester.strategies.base import FactorStrategy


@dataclass(frozen=True)
class MomentumFactorStrategy(FactorStrategy):
    quantiles: int = 5
    direction: str = "high_momentum"
    long_short: bool = False
    rebalance: str = "M"
    gross_exposure: float = 1.0

    def generate_weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        if self.direction not in {"high_momentum", "low_momentum"}:
            raise ValueError("direction must be high_momentum or low_momentum")
        if self.rebalance not in {"D", "W", "M", "Q"}:
            raise ValueError("rebalance must be D, W, M, or Q")

        groups = quantile_groups(factor, self.quantiles)
        high_group = self.quantiles - 1
        long_group = high_group if self.direction == "high_momentum" else 0
        short_group = 0 if self.direction == "high_momentum" else high_group

        weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
        for date in factor.index:
            long_mask = groups.loc[date].eq(long_group)
            short_mask = groups.loc[date].eq(short_group) if self.long_short else None
            weights.loc[date] = equal_weight_selection(
                long_mask,
                short_mask,
                self.gross_exposure,
            )

        if self.rebalance == "D":
            return weights

        periods = factor.index.to_period(self.rebalance)
        rebalance_dates = factor.groupby(periods).tail(1).index
        rebalanced = pd.DataFrame(float("nan"), index=factor.index, columns=factor.columns)
        rebalanced.loc[rebalance_dates] = weights.loc[rebalance_dates]
        return rebalanced.ffill().fillna(0.0)

