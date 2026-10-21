import pandas as pd
import pytest

from backtester.data.validator import validate_long_prices


def test_validator_rejects_duplicate_date_symbol_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.1],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_long_prices(frame)

