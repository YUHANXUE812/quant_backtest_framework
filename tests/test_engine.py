import pandas as pd
import pytest

from backtester.engine.backtest import BacktestEngine
from backtester.execution.broker import SimulatedBroker
from backtester.execution.costs import TransactionCostModel


def test_engine_applies_close_signal_on_next_trading_day() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.50, 0.10, -0.05]}, index=dates)
    target_weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    engine = BacktestEngine(
        initial_capital=1_000.0,
        broker=SimulatedBroker(TransactionCostModel()),
    )

    result = engine.run(returns, target_weights)

    assert result.ledger.loc[dates[0], "net_return"] == 0.0
    assert result.ledger.loc[dates[1], "net_return"] == pytest.approx(0.10)
    assert result.ledger.iloc[-1]["equity"] == pytest.approx(1_045.0)


def test_transaction_cost_model_charges_stamp_duty_on_sells_only() -> None:
    model = TransactionCostModel(
        commission_bps=3.0,
        sell_stamp_duty_bps=5.0,
        slippage_bps=5.0,
    )
    zero = pd.Series({"A": 0.0})
    invested = pd.Series({"A": 1.0})

    assert model.estimate(zero, invested) == pytest.approx(0.0008)
    assert model.estimate(invested, zero) == pytest.approx(0.0013)

