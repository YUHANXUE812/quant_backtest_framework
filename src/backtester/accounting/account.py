"""Account state updated from daily portfolio returns."""

import pandas as pd

from backtester.accounting.ledger import Ledger
from backtester.accounting.positions import PositionBook


class Account:
    def __init__(self, initial_capital: float, symbols: pd.Index) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)
        self.equity = float(initial_capital)
        self.positions = PositionBook(symbols)
        self.ledger = Ledger()

    def update(
        self,
        date: pd.Timestamp,
        target_weights: pd.Series,
        gross_return: float,
        cost_rate: float,
        turnover: float,
    ) -> None:
        net_return = gross_return - cost_rate
        self.equity *= 1.0 + net_return
        self.positions.set_target(target_weights)
        self.ledger.append(
            date=date,
            equity=self.equity,
            gross_return=gross_return,
            net_return=net_return,
            turnover=turnover,
            cost_rate=cost_rate,
        )

