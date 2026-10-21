"""Equal-weight quintile backtests for cross-sectional factor research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.signals.momentum_factors import FACTOR_NAMES

PORTFOLIO_ORDER = [
    "group_1",
    "group_2",
    "group_3",
    "group_4",
    "group_5",
    "baseline",
    "long_short",
    "hs300",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def prepare_layer_backtest_sample(
    processed_exposures: pd.DataFrame,
    monthly_sample: pd.DataFrame,
    stock_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] = FACTOR_NAMES,
) -> pd.DataFrame:
    """Attach next-first-day-to-month-end stock and HS300 returns to exposures."""
    _require_columns(
        processed_exposures,
        {"formation_date", "symbol", *factor_names},
        "Factor exposures",
    )
    _require_columns(
        monthly_sample,
        {"formation_date", "symbol", "next_trade_date", "label_end_date"},
        "Monthly sample",
    )
    _require_columns(stock_daily, {"date", "symbol", "close_adj"}, "Stock daily data")
    _require_columns(benchmark_daily, {"date", "close"}, "Benchmark daily data")

    exposures = processed_exposures.copy()
    sample = monthly_sample.copy()
    stocks = stock_daily.copy()
    benchmark = benchmark_daily.copy()
    exposures["formation_date"] = pd.to_datetime(exposures["formation_date"], errors="coerce")
    for column in ["formation_date", "next_trade_date", "label_end_date"]:
        sample[column] = pd.to_datetime(sample[column], errors="coerce")
    stocks["date"] = pd.to_datetime(stocks["date"], errors="coerce")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    stocks["close_adj"] = pd.to_numeric(stocks["close_adj"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    if "trade_status" in benchmark.columns:
        benchmark["trade_status"] = pd.to_numeric(benchmark["trade_status"], errors="coerce")
        benchmark = benchmark.loc[benchmark["trade_status"] == 1]
    benchmark = benchmark.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    calendar = pd.DatetimeIndex(benchmark["date"])

    prices = stocks.pivot_table(
        index="date", columns="symbol", values="close_adj", aggfunc="last"
    ).reindex(calendar).ffill()
    price_lookup = prices.stack(future_stack=True).dropna()
    benchmark_close = benchmark.set_index("date")["close"].reindex(calendar).ffill()
    merged = exposures.merge(
        sample.loc[:, ["formation_date", "symbol", "next_trade_date", "label_end_date"]],
        on=["formation_date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    start_index = pd.MultiIndex.from_arrays(
        [merged["next_trade_date"], merged["symbol"]], names=price_lookup.index.names
    )
    end_index = pd.MultiIndex.from_arrays(
        [merged["label_end_date"], merged["symbol"]], names=price_lookup.index.names
    )
    start_prices = price_lookup.reindex(start_index).to_numpy()
    end_prices = price_lookup.reindex(end_index).to_numpy()
    benchmark_start = benchmark_close.reindex(merged["next_trade_date"]).to_numpy()
    benchmark_end = benchmark_close.reindex(merged["label_end_date"]).to_numpy()
    merged["holding_period_return"] = end_prices / start_prices - 1.0
    merged["benchmark_holding_return"] = benchmark_end / benchmark_start - 1.0
    valid = (
        np.isfinite(merged["holding_period_return"])
        & np.isfinite(merged["benchmark_holding_return"])
    )
    return merged.loc[valid].reset_index(drop=True)


def run_quintile_backtests(
    prepared_sample: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] = FACTOR_NAMES,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Build equal-weight low-to-high factor portfolios and a low-minus-high spread."""
    _require_columns(
        prepared_sample,
        {
            "formation_date",
            "label_end_date",
            "symbol",
            "holding_period_return",
            "benchmark_holding_return",
            *factor_names,
        },
        "Prepared layer-backtest sample",
    )
    if quantiles < 2:
        raise ValueError("quantiles must be at least two")
    rows: list[dict[str, object]] = []
    for formation_date, monthly in prepared_sample.groupby("formation_date", sort=True):
        label_end_date = pd.Timestamp(monthly["label_end_date"].iloc[0])
        benchmark_return = float(monthly["benchmark_holding_return"].iloc[0])
        for factor in factor_names:
            values = monthly.loc[:, [factor, "holding_period_return"]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(values) < quantiles or values[factor].nunique() < 2:
                continue
            ranks = values[factor].rank(method="first")
            groups = pd.qcut(ranks, quantiles, labels=False) + 1
            group_returns = values.groupby(groups)["holding_period_return"].mean()
            if len(group_returns) != quantiles:
                continue
            for group in range(1, quantiles + 1):
                rows.append(
                    {
                        "formation_date": formation_date,
                        "label_end_date": label_end_date,
                        "factor": factor,
                        "portfolio": f"group_{group}",
                        "return": float(group_returns.loc[group]),
                    }
                )
            baseline_return = float(values["holding_period_return"].mean())
            rows.extend(
                [
                    {
                        "formation_date": formation_date,
                        "label_end_date": label_end_date,
                        "factor": factor,
                        "portfolio": "baseline",
                        "return": baseline_return,
                    },
                    {
                        "formation_date": formation_date,
                        "label_end_date": label_end_date,
                        "factor": factor,
                        "portfolio": "long_short",
                        "return": float(group_returns.loc[1] - group_returns.loc[quantiles]),
                    },
                    {
                        "formation_date": formation_date,
                        "label_end_date": label_end_date,
                        "factor": factor,
                        "portfolio": "hs300",
                        "return": benchmark_return,
                    },
                ]
            )
    return pd.DataFrame.from_records(rows)


def _annualized_return(returns: pd.Series, annualization: int) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    growth = float((1.0 + clean).prod())
    return growth ** (annualization / len(clean)) - 1.0 if growth > 0 else np.nan


def _max_drawdown(returns: pd.Series) -> float:
    net_value = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = net_value / net_value.cummax() - 1.0
    return float(-drawdown.min()) if not drawdown.empty else np.nan


def summarize_layer_performance(
    monthly_returns: pd.DataFrame,
    annualization: int = 12,
) -> pd.DataFrame:
    """Calculate absolute and baseline-relative metrics for every factor portfolio."""
    _require_columns(
        monthly_returns,
        {"label_end_date", "factor", "portfolio", "return"},
        "Monthly portfolio returns",
    )
    rows: list[dict[str, object]] = []
    for factor, factor_frame in monthly_returns.groupby("factor", sort=False):
        pivot = factor_frame.pivot(index="label_end_date", columns="portfolio", values="return")
        pivot = pivot.sort_index()
        baseline = pivot["baseline"]
        for portfolio in PORTFOLIO_ORDER:
            if portfolio not in pivot.columns:
                continue
            returns = pivot[portfolio].dropna()
            annualized_return = _annualized_return(returns, annualization)
            annualized_volatility = returns.std(ddof=1) * np.sqrt(annualization)
            row: dict[str, object] = {
                "factor": factor,
                "portfolio": portfolio,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": (
                    annualized_return / annualized_volatility
                    if annualized_volatility > 0
                    else np.nan
                ),
                "max_drawdown": _max_drawdown(returns),
                "annualized_excess_return": np.nan,
                "excess_annualized_volatility": np.nan,
                "information_ratio": np.nan,
                "relative_monthly_win_ratio": np.nan,
                "excess_max_drawdown": np.nan,
                "months": len(returns),
            }
            if portfolio.startswith("group_"):
                aligned = pd.concat([returns, baseline], axis=1, join="inner").dropna()
                relative_returns = (1.0 + aligned.iloc[:, 0]) / (1.0 + aligned.iloc[:, 1]) - 1.0
                excess_return = _annualized_return(relative_returns, annualization)
                excess_volatility = relative_returns.std(ddof=1) * np.sqrt(annualization)
                row.update(
                    {
                        "annualized_excess_return": excess_return,
                        "excess_annualized_volatility": excess_volatility,
                        "information_ratio": (
                            excess_return / excess_volatility
                            if excess_volatility > 0
                            else np.nan
                        ),
                        "relative_monthly_win_ratio": (
                            aligned.iloc[:, 0] > aligned.iloc[:, 1]
                        ).mean(),
                        "excess_max_drawdown": _max_drawdown(relative_returns),
                    }
                )
            rows.append(row)
    return pd.DataFrame.from_records(rows)
