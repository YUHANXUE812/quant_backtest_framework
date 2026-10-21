import pandas as pd
import pytest

from backtester.strategies.momentum import MomentumFactorStrategy


def test_high_momentum_strategy_selects_top_quantile() -> None:
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        index=pd.to_datetime(["2024-01-31"]),
        columns=list("ABCDE"),
    )
    strategy = MomentumFactorStrategy(quantiles=5, rebalance="D")

    weights = strategy.generate_weights(factor)

    assert weights.loc[factor.index[0], "E"] == pytest.approx(1.0)
    assert weights.loc[factor.index[0]].sum() == pytest.approx(1.0)
    assert (weights.loc[factor.index[0], list("ABCD")] == 0.0).all()


def test_monthly_strategy_only_changes_on_last_observation_of_month() -> None:
    factor = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]),
        columns=list("ABCDE"),
    )
    strategy = MomentumFactorStrategy(quantiles=5, rebalance="M")

    weights = strategy.generate_weights(factor)

    assert weights.loc["2024-01-30"].sum() == 0.0
    assert weights.loc["2024-01-31", "A"] == pytest.approx(1.0)
    assert weights.loc["2024-02-01", "E"] == pytest.approx(1.0)

