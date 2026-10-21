import pandas as pd
import pytest

from backtester.analytics.factor_ic import (
    calculate_monthly_factor_ic,
    cumulative_factor_ic,
    summarize_factor_ic,
)


def test_ic_summary_and_cumulative_curve_follow_monthly_correlations() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    factor_values = [-1.0, 0.0, 0.0, 1.0]
    return_patterns = [
        [-1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, -1.0, 0.0],
    ]
    exposures = []
    returns = []
    for current_date, monthly_returns in zip(dates, return_patterns, strict=True):
        for index, (factor, forward_return) in enumerate(
            zip(factor_values, monthly_returns, strict=True)
        ):
            symbol = f"S{index}"
            exposures.append(
                {"formation_date": current_date, "symbol": symbol, "factor": factor}
            )
            returns.append(
                {
                    "formation_date": current_date,
                    "symbol": symbol,
                    "next_month_return": forward_return,
                }
            )

    monthly = calculate_monthly_factor_ic(
        pd.DataFrame(exposures),
        pd.DataFrame(returns),
        factor_names=["factor"],
        minimum_observations=4,
    )
    summary = summarize_factor_ic(monthly, factor_names=["factor"])
    cumulative = cumulative_factor_ic(monthly, factor_names=["factor"])

    assert monthly["ic"].tolist() == pytest.approx([1.0, -1.0, 0.0])
    assert summary.loc[0, "mean_ic"] == pytest.approx(0.0)
    assert summary.loc[0, "std_ic"] == pytest.approx(1.0)
    assert summary.loc[0, "ir"] == pytest.approx(0.0)
    assert summary.loc[0, "positive_ic_ratio"] == pytest.approx(1.0 / 3.0)
    assert summary.loc[0, "abs_ic_gt_0_02_ratio"] == pytest.approx(2.0 / 3.0)
    assert cumulative["factor"].tolist() == pytest.approx([1.0, 0.0, 0.0])
