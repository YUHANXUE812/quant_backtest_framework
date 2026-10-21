"""Strategy interface."""

from abc import ABC, abstractmethod

import pandas as pd


class FactorStrategy(ABC):
    @abstractmethod
    def generate_weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        """Convert a date-by-asset factor panel to target portfolio weights."""

