"""Portfolio performance metrics."""

import math

import pandas as pd


def calculate_metrics(
    equity: pd.Series,
    returns: pd.Series,
    initial_capital: float,
    annualization: int = 252,
) -> dict[str, float | int]:
    if equity.empty:
        raise ValueError("Cannot calculate metrics for an empty equity curve")
    returns = returns.dropna()
    total_return = equity.iloc[-1] / initial_capital - 1.0
    periods = max(len(returns), 1)
    annualized_return = (1.0 + total_return) ** (annualization / periods) - 1.0
    volatility = returns.std(ddof=1) * math.sqrt(annualization) if len(returns) > 1 else 0.0
    standard_deviation = returns.std(ddof=1)
    sharpe = (
        returns.mean() / standard_deviation * math.sqrt(annualization)
        if len(returns) > 1 and standard_deviation > 0
        else 0.0
    )
    drawdown = equity / equity.cummax().clip(lower=initial_capital) - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "calmar_ratio": float(calmar),
        "max_drawdown": max_drawdown,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "average_daily_turnover": 0.0,
        "final_equity": float(equity.iloc[-1]),
    }

