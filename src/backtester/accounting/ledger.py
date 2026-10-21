"""Daily account ledger."""

import pandas as pd


class Ledger:
    def __init__(self) -> None:
        self._records: list[dict[str, object]] = []

    def append(
        self,
        date: pd.Timestamp,
        equity: float,
        gross_return: float,
        net_return: float,
        turnover: float,
        cost_rate: float,
    ) -> None:
        self._records.append(
            {
                "date": date,
                "equity": equity,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "cost_rate": cost_rate,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(
                columns=["equity", "gross_return", "net_return", "turnover", "cost_rate"]
            )
        return pd.DataFrame(self._records).set_index("date")

