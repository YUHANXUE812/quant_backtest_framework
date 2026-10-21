import pandas as pd
import pytest

from backtester.analytics.factor_layer_backtest import (
    prepare_layer_backtest_sample,
    run_quintile_backtests,
)


def test_prepare_sample_uses_next_trading_day_close() -> None:
    exposures = pd.DataFrame(
        {
            "formation_date": ["2024-01-31", "2024-01-31"],
            "symbol": ["A", "B"],
            "factor": [-1.0, 1.0],
        }
    )
    sample = pd.DataFrame(
        {
            "formation_date": ["2024-01-31", "2024-01-31"],
            "symbol": ["A", "B"],
            "next_trade_date": ["2024-02-01", "2024-02-01"],
            "label_end_date": ["2024-02-29", "2024-02-29"],
        }
    )
    stocks = pd.DataFrame(
        {
            "date": ["2024-02-01", "2024-02-01", "2024-02-29", "2024-02-29"],
            "symbol": ["A", "B", "A", "B"],
            "close_adj": [10.0, 20.0, 11.0, 18.0],
        }
    )
    benchmark = pd.DataFrame(
        {"date": ["2024-02-01", "2024-02-29"], "close": [100.0, 105.0], "trade_status": 1}
    )

    prepared = prepare_layer_backtest_sample(
        exposures, sample, stocks, benchmark, factor_names=["factor"]
    )

    assert prepared["holding_period_return"].tolist() == pytest.approx([0.10, -0.10])
    assert prepared["benchmark_holding_return"].tolist() == pytest.approx([0.05, 0.05])


def test_quintile_backtest_is_low_to_high_and_low_minus_high() -> None:
    prepared = pd.DataFrame(
        {
            "formation_date": "2024-01-31",
            "label_end_date": "2024-02-29",
            "symbol": [f"S{index}" for index in range(10)],
            "factor": list(range(10)),
            "holding_period_return": [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
            "benchmark_holding_return": 0.03,
        }
    )

    result = run_quintile_backtests(prepared, factor_names=["factor"])
    returns = result.set_index("portfolio")["return"]

    assert returns["group_1"] == pytest.approx(0.095)
    assert returns["group_5"] == pytest.approx(0.015)
    assert returns["baseline"] == pytest.approx(0.055)
    assert returns["long_short"] == pytest.approx(0.08)
    assert returns["hs300"] == pytest.approx(0.03)
