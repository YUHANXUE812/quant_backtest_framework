import pandas as pd
import pytest

from backtester.data.dataset import DatasetBuildConfig, build_monthly_momentum_dataset
from backtester.signals.transforms import mad_winsorize_series


def test_mad_winsorization_uses_median_absolute_deviation() -> None:
    values = pd.Series([0.0, 1.0, 2.0, 100.0])

    result = mad_winsorize_series(values, multiplier=5.0)

    assert result.tolist() == [0.0, 1.0, 2.0, 6.5]


def test_dataset_excludes_st_and_next_day_suspension() -> None:
    dates = pd.to_datetime(
        ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-28", "2024-02-29", "2024-03-01"]
    )
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 101.0, 102.0, 102.5, 104.0, 105.0, 106.0],
            "trade_status": 1,
        }
    )
    price_paths = {
        "000001.SZ": [10.0, 10.5, 11.0, 11.2, 12.8, 13.0, 13.2],
        "000002.SZ": [20.0, 20.5, 21.0, 21.1, 21.8, 22.0, 22.2],
        "000003.SZ": [30.0, 30.5, 31.0, 31.0, 31.8, 32.0, 32.2],
    }
    rows = []
    for symbol, prices in price_paths.items():
        for current_date, close in zip(dates, prices, strict=True):
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "close_adj": close,
                    "trade_status": 0 if symbol == "000003.SZ" and current_date == pd.Timestamp("2024-02-01") else 1,
                    "is_st": 1 if symbol == "000002.SZ" and current_date == pd.Timestamp("2024-01-31") else 0,
                    "is_pt": 0,
                    "float_market_cap": 1_000_000_000.0,
                    "pt_filter_quality": "unavailable_free_source",
                }
            )
    stocks = pd.DataFrame(rows)
    members = pd.DataFrame(
        {
            "symbol": list(price_paths),
            "in_date": "2025-01-01",
            "out_date": pd.NaT,
            "membership_quality": "current_constituents_only",
        }
    )
    config = DatasetBuildConfig(
        industry_code="801080",
        start_date="2024-01-31",
        end_date="2024-01-31",
        factor_lookback=2,
        factor_skip_recent=0,
        membership_mode="current_constituents",
    )

    samples, audit = build_monthly_momentum_dataset(stocks, benchmark, members, config)

    assert samples["symbol"].tolist() == ["000001.SZ"]
    assert samples.loc[0, "momentum_raw"] == pytest.approx(0.1)
    assert samples.loc[0, "momentum_zscore"] == 0.0
    expected_excess = (13.0 / 11.0 - 1.0) - (105.0 / 102.0 - 1.0)
    assert samples.loc[0, "next_month_excess_return"] == pytest.approx(expected_excess)
    assert audit.loc[0, "excluded_st_pt"] == 1
    assert audit.loc[0, "excluded_next_day_suspended_or_missing"] == 1
    assert audit.loc[0, "final_count"] == 1
