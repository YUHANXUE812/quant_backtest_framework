"""Time and asset return attribution."""

import pandas as pd


def annual_returns(returns: pd.Series) -> pd.Series:
    return returns.groupby(returns.index.year).apply(lambda values: (1.0 + values).prod() - 1.0)


def asset_contributions(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    weights, returns = weights.align(returns, join="inner", axis=0)
    weights, returns = weights.align(returns, join="inner", axis=1)
    return (weights * returns.fillna(0.0)).sum(axis=0).sort_values(ascending=False)

