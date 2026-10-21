"""Monthly information-coefficient diagnostics for factor research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.signals.momentum_factors import FACTOR_NAMES


def calculate_monthly_factor_ic(
    processed_exposures: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] = FACTOR_NAMES,
    return_column: str = "next_month_return",
    minimum_observations: int = 10,
) -> pd.DataFrame:
    """Calculate each factor's monthly cross-sectional Pearson IC."""
    exposure_required = {"formation_date", "symbol", *factor_names}
    return_required = {"formation_date", "symbol", return_column}
    missing_exposure = exposure_required - set(processed_exposures.columns)
    missing_return = return_required - set(forward_returns.columns)
    if missing_exposure:
        raise ValueError(f"Factor exposures are missing columns: {sorted(missing_exposure)}")
    if missing_return:
        raise ValueError(f"Forward returns are missing columns: {sorted(missing_return)}")
    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least three")

    exposures = processed_exposures.copy()
    returns = forward_returns.copy()
    exposures["formation_date"] = pd.to_datetime(exposures["formation_date"], errors="coerce")
    returns["formation_date"] = pd.to_datetime(returns["formation_date"], errors="coerce")
    merged = exposures.merge(
        returns.loc[:, ["formation_date", "symbol", return_column]],
        on=["formation_date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    merged[return_column] = pd.to_numeric(merged[return_column], errors="coerce")

    rows: list[dict[str, object]] = []
    for formation_date, monthly in merged.groupby("formation_date", sort=True):
        for factor in factor_names:
            values = monthly.loc[:, [factor, return_column]].apply(pd.to_numeric, errors="coerce")
            values = values.dropna()
            if (
                len(values) < minimum_observations
                or values[factor].nunique() < 2
                or values[return_column].nunique() < 2
            ):
                continue
            rows.append(
                {
                    "formation_date": formation_date,
                    "factor": factor,
                    "observations": len(values),
                    "ic": values[factor].corr(values[return_column], method="pearson"),
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_factor_ic(
    monthly_ic: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] = FACTOR_NAMES,
) -> pd.DataFrame:
    """Summarize IC mean, stability, IR, direction, and significance frequency."""
    required = {"factor", "ic"}
    missing = required - set(monthly_ic.columns)
    if missing:
        raise ValueError(f"Monthly IC data are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for factor in factor_names:
        values = pd.to_numeric(
            monthly_ic.loc[monthly_ic["factor"] == factor, "ic"], errors="coerce"
        ).dropna()
        standard_deviation = values.std(ddof=1)
        mean = values.mean()
        rows.append(
            {
                "factor": factor,
                "mean_ic": mean,
                "std_ic": standard_deviation,
                "ir": abs(mean) / standard_deviation if standard_deviation > 0 else np.nan,
                "positive_ic_ratio": values.gt(0).mean(),
                "abs_ic_gt_0_02_ratio": values.abs().gt(0.02).mean(),
                "available_months": len(values),
            }
        )
    return pd.DataFrame.from_records(rows)


def cumulative_factor_ic(
    monthly_ic: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] = FACTOR_NAMES,
) -> pd.DataFrame:
    """Return a wide cumulative-sum IC curve, treating unavailable months as zero."""
    required = {"formation_date", "factor", "ic"}
    missing = required - set(monthly_ic.columns)
    if missing:
        raise ValueError(f"Monthly IC data are missing columns: {sorted(missing)}")
    frame = monthly_ic.copy()
    frame["formation_date"] = pd.to_datetime(frame["formation_date"], errors="coerce")
    wide = frame.pivot(index="formation_date", columns="factor", values="ic")
    return wide.reindex(columns=factor_names).sort_index().fillna(0.0).cumsum()
