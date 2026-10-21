"""Simple portfolio-allocation rules."""

import pandas as pd


def equal_weight_selection(
    long_mask: pd.Series,
    short_mask: pd.Series | None,
    gross_exposure: float,
) -> pd.Series:
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive")
    weights = pd.Series(0.0, index=long_mask.index)
    long_count = int(long_mask.sum())
    if long_count == 0:
        return weights

    if short_mask is None:
        weights.loc[long_mask] = gross_exposure / long_count
        return weights

    short_count = int(short_mask.sum())
    if short_count == 0:
        return weights
    weights.loc[long_mask] = gross_exposure / 2.0 / long_count
    weights.loc[short_mask] = -gross_exposure / 2.0 / short_count
    return weights

