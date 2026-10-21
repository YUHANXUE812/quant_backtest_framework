"""Walk-forward splits and generic parameter-sensitivity tables."""

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd


def walk_forward_splits(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[tuple[pd.Index, pd.Index]]:
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")

    splits: list[tuple[pd.Index, pd.Index]] = []
    for start in range(0, len(index) - train_size - test_size + 1, step):
        train = index[start : start + train_size]
        test = index[start + train_size : start + train_size + test_size]
        splits.append((train, test))
    return splits


def parameter_sensitivity(
    values: Iterable[Any],
    runner: Callable[[Any], dict[str, float | int]],
    parameter_name: str,
) -> pd.DataFrame:
    rows = [{parameter_name: value, **runner(value)} for value in values]
    return pd.DataFrame(rows).set_index(parameter_name)

