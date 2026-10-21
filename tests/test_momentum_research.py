import numpy as np
import pandas as pd
import pytest

from backtester.analytics.factor_correlation import summarize_factor_correlations
from backtester.signals.momentum_factors import build_momentum_factor_exposures


def test_factor_formulas_include_halpha_and_weighted_returns() -> None:
    dates = pd.date_range("2010-01-31", periods=61, freq="ME")
    market_returns = np.linspace(-0.02, 0.03, 60)
    stock_returns = 0.01 + 2.0 * market_returns
    market_prices = 100.0 * np.cumprod(np.r_[1.0, 1.0 + market_returns])
    stock_prices = 20.0 * np.cumprod(np.r_[1.0, 1.0 + stock_returns])
    stocks = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001.SZ",
            "close_adj": stock_prices,
            "turnover_rate": 1.0,
        }
    )
    market = pd.DataFrame(
        {"date": dates, "close": market_prices, "trade_status": 1}
    )
    universe = pd.DataFrame(
        {"formation_date": [dates[-1]], "symbol": ["000001.SZ"]}
    )

    raw, processed = build_momentum_factor_exposures(stocks, market, universe)

    assert raw.loc[0, "HAlpha"] == pytest.approx(0.01)
    assert raw.loc[0, "return_1m"] == pytest.approx(stock_returns[-1])
    assert raw.loc[0, "wgt_return_1m"] == pytest.approx(stock_returns[-1])
    assert raw.loc[0, "exp_wgt_return_1m"] == pytest.approx(stock_returns[-1])
    assert processed.loc[0, "HAlpha"] == 0.0


def test_correlation_tables_use_monthly_cross_sections() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    factor_a = [-1.0, 0.0, 0.0, 1.0]
    factor_b = [
        [-1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, -1.0, 0.0],
    ]
    rows = []
    for current_date, values_b in zip(dates, factor_b, strict=True):
        for index, (value_a, value_b) in enumerate(zip(factor_a, values_b, strict=True)):
            rows.append(
                {
                    "formation_date": current_date,
                    "symbol": f"S{index}",
                    "A": value_a,
                    "B": value_b,
                }
            )

    mean, std, strength, monthly = summarize_factor_correlations(
        pd.DataFrame(rows), factor_names=["A", "B"], window_months=3
    )

    assert mean.loc["A", "B"] == pytest.approx(0.0)
    assert std.loc["A", "B"] == pytest.approx(1.0)
    assert strength.loc["A", "B"] == pytest.approx(0.0)
    assert len(monthly) == 12
