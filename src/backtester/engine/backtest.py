"""Vectorized-date loop for cross-sectional factor portfolios."""

from dataclasses import dataclass

import pandas as pd

from backtester.accounting.account import Account
from backtester.analytics.metrics import calculate_metrics
from backtester.execution.broker import SimulatedBroker


@dataclass(frozen=True)
class BacktestResult:
    ledger: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, float | int]


class BacktestEngine:
    """Apply close-derived target weights on the following trading day."""

    def __init__(
        self,
        initial_capital: float,
        broker: SimulatedBroker,
        annualization: int = 252,
    ) -> None:
        if annualization <= 0:
            raise ValueError("annualization must be positive")
        self.initial_capital = float(initial_capital)
        self.broker = broker
        self.annualization = annualization

    def run(self, returns: pd.DataFrame, target_weights: pd.DataFrame) -> BacktestResult:
        returns, target_weights = returns.align(target_weights, join="inner", axis=0)
        returns, target_weights = returns.align(target_weights, join="inner", axis=1)
        if returns.empty:
            raise ValueError("Returns and target weights have no overlapping data")

        applied_weights = target_weights.shift(1).fillna(0.0)
        account = Account(self.initial_capital, returns.columns)

        for date in returns.index:
            target = applied_weights.loc[date]
            _, turnover, cost_rate = self.broker.rebalance(account.positions.weights, target)
            gross_return = float((target * returns.loc[date].fillna(0.0)).sum())
            account.update(date, target, gross_return, cost_rate, turnover)

        ledger = account.ledger.to_frame()
        metrics = calculate_metrics(
            equity=ledger["equity"],
            returns=ledger["net_return"],
            initial_capital=self.initial_capital,
            annualization=self.annualization,
        )
        metrics["average_daily_turnover"] = float(ledger["turnover"].mean())
        return BacktestResult(ledger=ledger, weights=applied_weights, metrics=metrics)

