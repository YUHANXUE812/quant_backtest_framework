"""Momentum-factor calculations."""

import pandas as pd


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate close-to-close returns without implicitly filling missing prices."""
    return prices.pct_change(fill_method=None)


def price_momentum(
    prices: pd.DataFrame,
    lookback: int,
    skip_recent: int = 0,
) -> pd.DataFrame:
    """Return P(t-skip) / P(t-lookback) - 1 for every asset and date."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip_recent < 0:
        raise ValueError("skip_recent cannot be negative")
    if skip_recent >= lookback:
        raise ValueError("skip_recent must be smaller than lookback")
    return prices.shift(skip_recent) / prices.shift(lookback) - 1.0

