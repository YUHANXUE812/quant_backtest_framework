"""Download free raw data for the Shenwan Electronics momentum study."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtester.config import load_config_directory
from backtester.data.free_sources import download_electronics_members, download_free_raw_data


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free A-share research data")
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument(
        "--members-only",
        action="store_true",
        help="Only refresh the current Shenwan industry membership file",
    )
    args = parser.parse_args()
    dataset = load_config_directory(args.config_dir)["dataset"]

    members = download_electronics_members(str(dataset["industry_code"]))
    member_path = PROJECT_ROOT / dataset["raw_membership_file"]
    _save(members, member_path)
    print(f"Saved {len(members)} current constituents to {member_path}")
    if args.members_only:
        return

    lookback = int(dataset["factor_lookback"]) + int(dataset["factor_skip_recent"])
    download_start = pd.Timestamp(dataset["start_date"]) - pd.Timedelta(days=lookback * 2 + 45)
    download_end = pd.Timestamp(dataset["end_date"]) + pd.Timedelta(days=45)
    stocks, benchmark, failures = download_free_raw_data(
        members,
        str(dataset["benchmark_code"]),
        download_start,
        download_end,
    )
    stock_path = PROJECT_ROOT / dataset["raw_stock_daily_file"]
    benchmark_path = PROJECT_ROOT / dataset["raw_benchmark_file"]
    failure_path = PROJECT_ROOT / dataset["download_failures_file"]
    _save(stocks, stock_path)
    _save(benchmark, benchmark_path)
    _save(failures, failure_path)
    print(f"Saved {len(stocks)} stock-day rows to {stock_path}")
    print(f"Saved {len(benchmark)} benchmark rows to {benchmark_path}")
    print(f"Recorded {len(failures)} download failures in {failure_path}")


if __name__ == "__main__":
    main()
