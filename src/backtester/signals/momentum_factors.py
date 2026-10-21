"""Construction of the momentum-factor family used in cross-sectional research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.signals.transforms import mad_winsorize_series, standardize_series

FACTOR_WINDOWS = (1, 3, 6, 12)
FACTOR_NAMES = [
    "HAlpha",
    *(f"return_{months}m" for months in FACTOR_WINDOWS),
    *(f"wgt_return_{months}m" for months in FACTOR_WINDOWS),
    *(f"exp_wgt_return_{months}m" for months in FACTOR_WINDOWS),
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _month_ends(calendar: pd.DatetimeIndex) -> pd.Series:
    values = pd.Series(calendar, index=calendar)
    return values.groupby(calendar.to_period("M")).max()


def _halpha(
    monthly_stock_returns: pd.DataFrame,
    monthly_market_returns: pd.Series,
    lookback: int,
) -> pd.DataFrame:
    mean_stock = monthly_stock_returns.rolling(lookback, min_periods=lookback).mean()
    mean_market = monthly_market_returns.rolling(lookback, min_periods=lookback).mean()
    mean_product = monthly_stock_returns.mul(monthly_market_returns, axis=0).rolling(
        lookback, min_periods=lookback
    ).mean()
    mean_market_squared = monthly_market_returns.pow(2).rolling(
        lookback, min_periods=lookback
    ).mean()
    covariance = mean_product.sub(mean_stock.mul(mean_market, axis=0))
    market_variance = mean_market_squared - mean_market.pow(2)
    beta = covariance.div(market_variance.replace(0.0, np.nan), axis=0)
    return mean_stock - beta.mul(mean_market, axis=0)


def _weighted_daily_factors(
    daily_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    month_ends: pd.Series,
    windows: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    calendar = daily_returns.index
    columns = daily_returns.columns
    for months in windows:
        weighted = pd.DataFrame(np.nan, index=month_ends.to_numpy(), columns=columns)
        exponential = weighted.copy()
        for period, formation_date in month_ends.items():
            previous_period = period - months
            if previous_period not in month_ends.index:
                continue
            start_date = month_ends.loc[previous_period]
            window_dates = calendar[(calendar > start_date) & (calendar <= formation_date)]
            if window_dates.empty:
                continue
            returns_window = daily_returns.loc[window_dates]
            turnover_window = turnover.loc[window_dates].where(returns_window.notna())
            denominator = turnover_window.sum(axis=0, min_count=1).replace(0.0, np.nan)
            weighted.loc[formation_date] = (
                returns_window.mul(turnover_window).sum(axis=0, min_count=1) / denominator
            )

            trading_day_distance = np.arange(len(window_dates) - 1, -1, -1, dtype=float)
            decay = np.exp(-trading_day_distance / (4.0 * months))
            exp_weights = turnover_window.mul(decay, axis=0)
            exp_denominator = exp_weights.sum(axis=0, min_count=1).replace(0.0, np.nan)
            exponential.loc[formation_date] = (
                returns_window.mul(exp_weights).sum(axis=0, min_count=1) / exp_denominator
            )
        output[f"wgt_return_{months}m"] = weighted
        output[f"exp_wgt_return_{months}m"] = exponential
    return output


def build_momentum_factor_exposures(
    stock_daily: pd.DataFrame,
    market_daily: pd.DataFrame,
    universe: pd.DataFrame,
    windows: tuple[int, ...] = FACTOR_WINDOWS,
    halpha_lookback: int = 60,
    mad_multiplier: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw and processed factor exposures for the requested monthly universe."""
    _require_columns(
        stock_daily,
        {"date", "symbol", "close_adj", "turnover_rate"},
        "Stock daily data",
    )
    _require_columns(market_daily, {"date", "close"}, "Market index data")
    _require_columns(universe, {"formation_date", "symbol"}, "Research universe")
    if not windows or any(months <= 0 for months in windows):
        raise ValueError("Factor windows must contain positive month counts")
    if halpha_lookback <= 1:
        raise ValueError("HAlpha lookback must be greater than one month")

    stocks = stock_daily.copy()
    market = market_daily.copy()
    sample = universe.loc[:, ["formation_date", "symbol"]].drop_duplicates().copy()
    stocks["date"] = pd.to_datetime(stocks["date"], errors="coerce")
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    sample["formation_date"] = pd.to_datetime(sample["formation_date"], errors="coerce")
    stocks["close_adj"] = pd.to_numeric(stocks["close_adj"], errors="coerce")
    stocks["turnover_rate"] = pd.to_numeric(stocks["turnover_rate"], errors="coerce")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    if "trade_status" in market.columns:
        market["trade_status"] = pd.to_numeric(market["trade_status"], errors="coerce")
        market = market.loc[market["trade_status"] == 1]
    market = market.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    calendar = pd.DatetimeIndex(market["date"])
    month_ends = _month_ends(calendar)

    stocks = stocks.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])
    prices = stocks.pivot_table(
        index="date", columns="symbol", values="close_adj", aggfunc="last"
    ).reindex(calendar).ffill()
    turnover = stocks.pivot_table(
        index="date", columns="symbol", values="turnover_rate", aggfunc="last"
    ).reindex(index=calendar, columns=prices.columns).fillna(0.0)
    daily_returns = prices.pct_change(fill_method=None)
    monthly_prices = prices.reindex(month_ends.to_numpy())
    monthly_stock_returns = monthly_prices.pct_change(fill_method=None)
    market_close = market.set_index("date")["close"].reindex(calendar)
    monthly_market_returns = market_close.reindex(month_ends.to_numpy()).pct_change(fill_method=None)

    factor_panels: dict[str, pd.DataFrame] = {
        "HAlpha": _halpha(monthly_stock_returns, monthly_market_returns, halpha_lookback)
    }
    for months in windows:
        factor_panels[f"return_{months}m"] = monthly_prices / monthly_prices.shift(months) - 1.0
    factor_panels.update(_weighted_daily_factors(daily_returns, turnover, month_ends, windows))
    factor_names = [
        "HAlpha",
        *(f"return_{months}m" for months in windows),
        *(f"wgt_return_{months}m" for months in windows),
        *(f"exp_wgt_return_{months}m" for months in windows),
    ]

    raw_parts: list[pd.DataFrame] = []
    for formation_date, cross_section in sample.groupby("formation_date", sort=True):
        symbols = pd.Index(cross_section["symbol"].astype(str))
        part = pd.DataFrame({"formation_date": formation_date, "symbol": symbols})
        for name in factor_names:
            panel = factor_panels[name]
            if formation_date in panel.index:
                part[name] = panel.reindex(index=[formation_date], columns=symbols).iloc[0].to_numpy()
            else:
                part[name] = np.nan
        raw_parts.append(part)
    raw = pd.concat(raw_parts, ignore_index=True)

    processed = raw.copy()
    for _, indices in processed.groupby("formation_date").groups.items():
        for name in factor_names:
            winsorized = mad_winsorize_series(raw.loc[indices, name], mad_multiplier)
            processed.loc[indices, name] = standardize_series(winsorized)
    return raw, processed
