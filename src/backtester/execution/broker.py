"""Target-weight rebalance simulation."""

import pandas as pd

from backtester.execution.costs import TransactionCostModel
from backtester.execution.orders import WeightOrder


class SimulatedBroker:
    def __init__(self, cost_model: TransactionCostModel) -> None:
        self.cost_model = cost_model

    def rebalance(
        self,
        current: pd.Series,
        target: pd.Series,
    ) -> tuple[list[WeightOrder], float, float]:
        current, target = current.align(target, join="outer", fill_value=0.0)
        changes = target - current
        orders = [
            WeightOrder(symbol, float(current[symbol]), float(target[symbol]))
            for symbol in changes.index[changes.ne(0.0)]
        ]
        turnover = float(changes.abs().sum())
        cost = self.cost_model.estimate(current, target)
        return orders, turnover, cost

