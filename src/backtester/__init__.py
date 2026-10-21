"""A-share momentum-factor research framework."""

from backtester.engine.backtest import BacktestEngine, BacktestResult
from backtester.strategies.momentum import MomentumFactorStrategy

__all__ = ["BacktestEngine", "BacktestResult", "MomentumFactorStrategy"]

