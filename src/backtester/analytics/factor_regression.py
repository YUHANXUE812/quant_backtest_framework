"""Monthly cross-sectional factor-return regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.signals.momentum_factors import FACTOR_NAMES


def _fit_single_factor_wls(
    exposure: pd.Series,
    target: pd.Series,
    weight: pd.Series,
) -> tuple[float, float, float]:
    x = np.column_stack([np.ones(len(exposure)), exposure.to_numpy(dtype=float)])
    y = target.to_numpy(dtype=float)
    w = weight.to_numpy(dtype=float)
    xtwx = x.T @ (w[:, None] * x)
    if np.linalg.matrix_rank(xtwx) < 2:
        return np.nan, np.nan, np.nan
    inverse = np.linalg.inv(xtwx)
    coefficients = inverse @ (x.T @ (w * y))
    residuals = y - x @ coefficients
    degrees_of_freedom = len(y) - x.shape[1]
    if degrees_of_freedom <= 0:
        return np.nan, np.nan, np.nan
    residual_variance = float(np.sum(w * residuals**2) / degrees_of_freedom)
    standard_error = float(np.sqrt(max(residual_variance * inverse[1, 1], 0.0)))
    factor_return = float(coefficients[1])
    t_value = factor_return / standard_error if standard_error > 0 else np.nan
    return factor_return, standard_error, t_value


def run_monthly_factor_regressions(
    processed_exposures: pd.DataFrame,
    regression_sample: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] = FACTOR_NAMES,
    target_column: str = "next_month_excess_return",
    weight_column: str = "wls_weight",
    minimum_observations: int = 10,
) -> pd.DataFrame:
    """Estimate one intercept-plus-factor WLS regression per factor and month."""
    exposure_required = {"formation_date", "symbol", *factor_names}
    sample_required = {"formation_date", "symbol", target_column, weight_column}
    missing_exposure = exposure_required - set(processed_exposures.columns)
    missing_sample = sample_required - set(regression_sample.columns)
    if missing_exposure:
        raise ValueError(f"Factor exposures are missing columns: {sorted(missing_exposure)}")
    if missing_sample:
        raise ValueError(f"Regression sample is missing columns: {sorted(missing_sample)}")
    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least three")

    exposures = processed_exposures.copy()
    sample = regression_sample.copy()
    exposures["formation_date"] = pd.to_datetime(exposures["formation_date"], errors="coerce")
    sample["formation_date"] = pd.to_datetime(sample["formation_date"], errors="coerce")
    merged = exposures.merge(
        sample.loc[:, ["formation_date", "symbol", target_column, weight_column]],
        on=["formation_date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    merged[target_column] = pd.to_numeric(merged[target_column], errors="coerce")
    merged[weight_column] = pd.to_numeric(merged[weight_column], errors="coerce")

    rows: list[dict[str, object]] = []
    for formation_date, monthly in merged.groupby("formation_date", sort=True):
        for factor in factor_names:
            values = monthly.loc[:, [factor, target_column, weight_column]].apply(
                pd.to_numeric, errors="coerce"
            )
            values = values.dropna()
            values = values.loc[values[weight_column] > 0]
            if len(values) < minimum_observations or values[factor].nunique() < 2:
                continue
            factor_return, standard_error, t_value = _fit_single_factor_wls(
                values[factor], values[target_column], values[weight_column]
            )
            rows.append(
                {
                    "formation_date": formation_date,
                    "factor": factor,
                    "observations": len(values),
                    "factor_return": factor_return,
                    "standard_error": standard_error,
                    "t_value": t_value,
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_factor_regressions(
    monthly_regressions: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] = FACTOR_NAMES,
) -> pd.DataFrame:
    """Build the six factor-evaluation statistics from monthly regressions."""
    required = {"factor", "factor_return", "t_value"}
    missing = required - set(monthly_regressions.columns)
    if missing:
        raise ValueError(f"Monthly regressions are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for factor in factor_names:
        sample = monthly_regressions.loc[monthly_regressions["factor"] == factor]
        t_values = pd.to_numeric(sample["t_value"], errors="coerce").dropna()
        returns = pd.to_numeric(sample["factor_return"], errors="coerce").dropna()
        t_std = t_values.std(ddof=1)
        return_std = returns.std(ddof=1)
        rows.append(
            {
                "factor": factor,
                "mean_abs_t": t_values.abs().mean(),
                "abs_t_gt_2_ratio": t_values.abs().gt(2.0).mean(),
                "mean_t": t_values.mean(),
                "mean_t_over_std": t_values.mean() / t_std if t_std > 0 else np.nan,
                "mean_factor_return": returns.mean(),
                "factor_return_series_t": (
                    returns.mean() / (return_std / np.sqrt(len(returns)))
                    if len(returns) > 1 and return_std > 0
                    else np.nan
                ),
                "available_months": len(sample),
            }
        )
    return pd.DataFrame.from_records(rows)
