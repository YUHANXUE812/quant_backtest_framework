"""Order data structures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightOrder:
    symbol: str
    current_weight: float
    target_weight: float

    @property
    def weight_change(self) -> float:
        return self.target_weight - self.current_weight

