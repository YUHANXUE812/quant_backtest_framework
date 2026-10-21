import numpy as np
import pandas as pd
import pytest

from backtester.analytics.factor_regression import (
    run_monthly_factor_regressions,
    summarize_factor_regressions,
)


def test_monthly_wls_recovers_factor_return_and_t_summary() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    exposures = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    noise = np.array([0.01, -0.02, 0.02, -0.02, 0.01])
    factor_rows = []
    sample_rows = []
    for current_date in dates:
        for index, exposure in enumerate(exposures):
            symbol = f"S{index}"
            factor_rows.append(
                {"formation_date": current_date, "symbol": symbol, "factor": exposure}
            )
            sample_rows.append(
                {
                    "formation_date": current_date,
                    "symbol": symbol,
                    "next_month_excess_return": 0.02 + 0.03 * exposure + noise[index],
                    "wls_weight": 1.0,
                }
            )

    monthly = run_monthly_factor_regressions(
        pd.DataFrame(factor_rows),
        pd.DataFrame(sample_rows),
        factor_names=["factor"],
        minimum_observations=5,
    )
    summary = summarize_factor_regressions(monthly, factor_names=["factor"])

    assert monthly["factor_return"].tolist() == pytest.approx([0.03, 0.03, 0.03])
    assert (monthly["t_value"].abs() > 2).all()
    assert summary.loc[0, "abs_t_gt_2_ratio"] == 1.0
    assert summary.loc[0, "mean_factor_return"] == pytest.approx(0.03)
