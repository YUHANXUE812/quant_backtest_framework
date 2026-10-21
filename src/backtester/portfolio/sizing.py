"""Volatility-targeting helpers."""

import numpy as np
import pandas as pd


def volatility_multiplier(
    portfolio_returns: pd.Series,
    target_annual_volatility: float,
    lookback: int,
    max_leverage: float,
    annualization: int = 252,
) -> pd.Series:
    if target_annual_volatility <= 0 or lookback <= 1 or max_leverage <= 0:
        raise ValueError("Volatility-targeting parameters must be positive")
    realized = portfolio_returns.rolling(lookback).std(ddof=1) * np.sqrt(annualization)
    return (target_annual_volatility / realized).clip(upper=max_leverage).shift(1).fillna(0.0)

