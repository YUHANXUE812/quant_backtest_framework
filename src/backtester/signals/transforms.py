"""Cross-sectional factor transformations."""

import numpy as np
import pandas as pd


def mad_winsorize_series(values: pd.Series, multiplier: float = 5.0) -> pd.Series:
    """Clip a cross-section at median +/- multiplier * median absolute deviation."""
    if multiplier <= 0:
        raise ValueError("MAD multiplier must be positive")
    valid = values.dropna()
    if valid.empty:
        return values.copy()
    median = valid.median()
    median_absolute_deviation = (valid - median).abs().median()
    if median_absolute_deviation == 0:
        return values.copy()
    lower = median - multiplier * median_absolute_deviation
    upper = median + multiplier * median_absolute_deviation
    return values.clip(lower, upper)


def standardize_series(values: pd.Series) -> pd.Series:
    """Cross-sectionally standardize and neutrally impute missing values with zero."""
    mean = values.mean()
    standard_deviation = values.std(ddof=0)
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)
    return ((values - mean) / standard_deviation).fillna(0.0)


def winsorize_cross_section(
    values: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    if not 0 <= lower < upper <= 1:
        raise ValueError("Winsorization bounds must satisfy 0 <= lower < upper <= 1")

    def clip_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if valid.empty:
            return row
        return row.clip(valid.quantile(lower), valid.quantile(upper))

    return values.apply(clip_row, axis=1)


def zscore_cross_section(values: pd.DataFrame) -> pd.DataFrame:
    means = values.mean(axis=1)
    standard_deviations = values.std(axis=1, ddof=0).replace(0.0, np.nan)
    return values.sub(means, axis=0).div(standard_deviations, axis=0)


def quantile_groups(values: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")

    def group_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        result = pd.Series(np.nan, index=row.index, dtype=float)
        if len(valid) < quantiles:
            return result
        ranks = valid.rank(method="first")
        result.loc[valid.index] = pd.qcut(ranks, quantiles, labels=False).astype(float)
        return result

    return values.apply(group_row, axis=1)

