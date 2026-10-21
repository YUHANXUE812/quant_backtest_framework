"""Correlation diagnostics for cross-sectional factor research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.signals.momentum_factors import FACTOR_NAMES


def summarize_factor_correlations(
    processed_exposures: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] = FACTOR_NAMES,
    window_months: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize monthly cross-sectional factor correlations over the latest window."""
    required = {"formation_date", "symbol", *factor_names}
    missing = required - set(processed_exposures.columns)
    if missing:
        raise ValueError(f"Processed factor exposures are missing columns: {sorted(missing)}")
    if window_months <= 1:
        raise ValueError("Correlation window must be greater than one month")
    frame = processed_exposures.copy()
    frame["formation_date"] = pd.to_datetime(frame["formation_date"], errors="coerce")
    dates = sorted(frame["formation_date"].dropna().unique())
    selected_dates = dates[-window_months:]
    if len(selected_dates) < window_months:
        raise ValueError(
            f"Need {window_months} monthly cross-sections, found {len(selected_dates)}"
        )

    matrices: list[pd.DataFrame] = []
    for formation_date in selected_dates:
        cross_section = frame.loc[frame["formation_date"] == formation_date, factor_names]
        matrices.append(cross_section.corr(method="pearson", min_periods=3))
    values = np.stack([matrix.to_numpy() for matrix in matrices])
    mean = pd.DataFrame(np.nanmean(values, axis=0), index=factor_names, columns=factor_names)
    std = pd.DataFrame(np.nanstd(values, axis=0, ddof=1), index=factor_names, columns=factor_names)
    strength = mean / std.replace(0.0, np.nan)
    for table in (mean, std, strength):
        np.fill_diagonal(table.values, np.nan)

    monthly = pd.concat(
        matrices,
        keys=pd.DatetimeIndex(selected_dates, name="formation_date"),
        names=["formation_date", "factor_a"],
    ).rename_axis(columns="factor_b").stack(future_stack=True).rename("correlation").reset_index()
    return mean, std, strength, monthly
