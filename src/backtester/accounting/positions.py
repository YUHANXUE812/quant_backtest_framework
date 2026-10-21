"""Current portfolio weights."""

import pandas as pd


class PositionBook:
    def __init__(self, symbols: pd.Index) -> None:
        self._weights = pd.Series(0.0, index=symbols)

    @property
    def weights(self) -> pd.Series:
        return self._weights.copy()

    def set_target(self, target: pd.Series) -> None:
        self._weights = target.reindex(self._weights.index, fill_value=0.0).astype(float)

