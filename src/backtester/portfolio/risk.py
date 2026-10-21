"""Portfolio-weight constraints."""

import pandas as pd


def apply_weight_limits(
    weights: pd.DataFrame,
    max_stock_weight: float,
    max_gross_exposure: float,
) -> pd.DataFrame:
    if max_stock_weight <= 0 or max_gross_exposure <= 0:
        raise ValueError("Weight limits must be positive")

    def constrain(row: pd.Series) -> pd.Series:
        limited = row.clip(-max_stock_weight, max_stock_weight)
        gross = limited.abs().sum()
        if gross > max_gross_exposure:
            limited *= max_gross_exposure / gross
        return limited

    return weights.apply(constrain, axis=1)

