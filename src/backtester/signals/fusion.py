"""Utilities for combining multiple aligned signals."""

import pandas as pd


def weighted_signal_average(
    signals: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    if not signals:
        raise ValueError("At least one signal is required")
    if set(signals) != set(weights):
        raise ValueError("Signal and weight names must match")
    total_weight = sum(abs(weight) for weight in weights.values())
    if total_weight == 0:
        raise ValueError("Signal weights cannot all be zero")

    combined: pd.DataFrame | None = None
    for name, signal in signals.items():
        contribution = signal * (weights[name] / total_weight)
        combined = contribution if combined is None else combined.add(contribution, fill_value=0.0)
    return combined

