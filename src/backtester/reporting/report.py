"""Persist tabular results and a compact equity/drawdown chart."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtester.analytics.attribution import annual_returns
from backtester.engine.backtest import BacktestResult


def save_report(result: BacktestResult, output_directory: str | Path) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    result.ledger.to_csv(output_directory / "ledger.csv")
    result.weights.to_csv(output_directory / "weights.csv")
    annual_returns(result.ledger["net_return"]).to_csv(
        output_directory / "annual_returns.csv",
        header=["return"],
    )
    (output_directory / "summary.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    equity = result.ledger["equity"]
    drawdown = equity / equity.cummax() - 1.0
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    equity.plot(ax=axes[0], title="Momentum portfolio equity")
    drawdown.plot(ax=axes[1], title="Drawdown")
    axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "performance.png", dpi=150)
    plt.close(figure)

