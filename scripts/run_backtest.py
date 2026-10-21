"""Run the configured A-share momentum-factor backtest."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtester.config import load_config_directory
from backtester.data.aligner import align_prices
from backtester.data.cleaner import clean_price_panel
from backtester.data.loader import load_benchmark, load_price_panel
from backtester.data.validator import validate_panel
from backtester.engine.backtest import BacktestEngine, BacktestResult
from backtester.execution.broker import SimulatedBroker
from backtester.execution.costs import TransactionCostModel
from backtester.portfolio.risk import apply_weight_limits
from backtester.portfolio.sizing import volatility_multiplier
from backtester.reporting.report import save_report
from backtester.signals.factors import calculate_returns, price_momentum
from backtester.signals.transforms import winsorize_cross_section, zscore_cross_section
from backtester.strategies.momentum import MomentumFactorStrategy


def run_research(
    config_directory: Path,
    lookback_override: int | None = None,
    skip_override: int | None = None,
    save_outputs: bool = True,
) -> BacktestResult:
    config = load_config_directory(config_directory)
    data_config = config["data"]
    factor_config = config["factor"]
    strategy_config = config["strategy"]
    portfolio_config = config["portfolio"]
    execution_config = config["execution"]
    backtest_config = config["backtest"]

    asset_path = PROJECT_ROOT / data_config["asset_prices"]
    benchmark_path = PROJECT_ROOT / data_config["benchmark_prices"]
    prices = clean_price_panel(load_price_panel(asset_path))
    benchmark = load_benchmark(benchmark_path)
    prices, benchmark = align_prices(prices, benchmark)

    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")
    if start_date:
        prices, benchmark = prices.loc[start_date:], benchmark.loc[start_date:]
    if end_date:
        prices, benchmark = prices.loc[:end_date], benchmark.loc[:end_date]

    lookback = lookback_override or int(factor_config["lookback"])
    skip_recent = skip_override if skip_override is not None else int(factor_config["skip_recent"])
    validate_panel(prices, lookback + 2)

    returns = calculate_returns(prices)
    momentum = price_momentum(prices, lookback, skip_recent)
    momentum = winsorize_cross_section(
        momentum,
        float(factor_config["winsor_lower"]),
        float(factor_config["winsor_upper"]),
    )
    signal = zscore_cross_section(momentum)

    strategy = MomentumFactorStrategy(
        quantiles=int(strategy_config["quantiles"]),
        direction=strategy_config["direction"],
        long_short=bool(strategy_config["long_short"]),
        rebalance=strategy_config["rebalance"],
        gross_exposure=float(portfolio_config["gross_exposure"]),
    )
    weights = strategy.generate_weights(signal)

    target_volatility = portfolio_config.get("target_annual_volatility")
    if target_volatility is not None:
        preliminary_returns = (weights.shift(1) * returns.fillna(0.0)).sum(axis=1)
        multiplier = volatility_multiplier(
            preliminary_returns,
            float(target_volatility),
            int(portfolio_config["volatility_lookback"]),
            float(portfolio_config["max_leverage"]),
            int(backtest_config["annualization"]),
        )
        weights = weights.mul(multiplier, axis=0)

    weights = apply_weight_limits(
        weights,
        float(portfolio_config["max_stock_weight"]),
        float(portfolio_config["max_gross_exposure"]),
    )
    cost_model = TransactionCostModel(
        commission_bps=float(execution_config["commission_bps"]),
        sell_stamp_duty_bps=float(execution_config["sell_stamp_duty_bps"]),
        slippage_bps=float(execution_config["slippage_bps"]),
    )
    engine = BacktestEngine(
        initial_capital=float(backtest_config["initial_capital"]),
        broker=SimulatedBroker(cost_model),
        annualization=int(backtest_config["annualization"]),
    )
    result = engine.run(returns, weights)
    if save_outputs:
        save_report(result, PROJECT_ROOT / "data" / "results")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an A-share momentum-factor backtest")
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument("--lookback", type=int)
    parser.add_argument("--skip-recent", type=int)
    args = parser.parse_args()
    result = run_research(args.config_dir, args.lookback, args.skip_recent)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

