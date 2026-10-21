"""Generate deterministic multi-asset prices for a smoke test."""

import csv
import math
from datetime import date, timedelta
from pathlib import Path


def trading_dates(start: date, count: int) -> list[date]:
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "raw"
    output.mkdir(parents=True, exist_ok=True)
    dates = trading_dates(date(2023, 1, 3), 420)
    benchmark_price = 100.0
    asset_prices = {f"A{number:03d}": 50.0 + number for number in range(1, 26)}
    benchmark_rows: list[dict[str, object]] = []
    asset_rows: list[dict[str, object]] = []

    for index, current_date in enumerate(dates):
        benchmark_return = 0.00015 + 0.006 * math.sin(index / 13.0)
        benchmark_price *= 1.0 + benchmark_return
        benchmark_rows.append(
            {"date": current_date.isoformat(), "close": f"{benchmark_price:.6f}"}
        )

        for number, (symbol, price) in enumerate(asset_prices.items(), start=1):
            market_loading = 0.6 + 0.035 * number
            persistent_component = 0.0018 * math.sin(index / 28.0 + number / 5.0)
            idiosyncratic_component = 0.0007 * math.sin(index / 3.0 + number)
            asset_return = (
                0.0001
                + market_loading * benchmark_return
                + persistent_component
                + idiosyncratic_component
            )
            price *= 1.0 + asset_return
            asset_prices[symbol] = price
            asset_rows.append(
                {
                    "date": current_date.isoformat(),
                    "symbol": symbol,
                    "close": f"{price:.6f}",
                }
            )

    with (output / "asset_prices.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["date", "symbol", "close"])
        writer.writeheader()
        writer.writerows(asset_rows)
    with (output / "benchmark_prices.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["date", "close"])
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"Wrote {len(asset_rows)} asset rows and {len(benchmark_rows)} benchmark rows")


if __name__ == "__main__":
    main()

