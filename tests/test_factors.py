import pandas as pd
import pytest

from backtester.signals.factors import price_momentum


def test_price_momentum_respects_lookback_and_skip_period() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 105.0, 110.0, 120.0, 126.0]},
        index=pd.date_range("2024-01-01", periods=5),
    )

    momentum = price_momentum(prices, lookback=3, skip_recent=1)

    assert momentum.iloc[-1, 0] == pytest.approx(120.0 / 105.0 - 1.0)


@pytest.mark.parametrize(
    ("lookback", "skip_recent"),
    [(0, 0), (5, -1), (5, 5)],
)
def test_price_momentum_rejects_invalid_windows(lookback: int, skip_recent: int) -> None:
    prices = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(ValueError):
        price_momentum(prices, lookback, skip_recent)

