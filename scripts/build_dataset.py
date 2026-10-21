"""Build the monthly Electronics momentum-regression dataset."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtester.config import load_config_directory
from backtester.data.dataset import DatasetBuildConfig, build_monthly_momentum_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build monthly momentum research samples")
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    args = parser.parse_args()
    dataset = load_config_directory(args.config_dir)["dataset"]

    members = pd.read_csv(PROJECT_ROOT / dataset["raw_membership_file"])
    stocks = pd.read_csv(PROJECT_ROOT / dataset["raw_stock_daily_file"])
    benchmark = pd.read_csv(PROJECT_ROOT / dataset["raw_benchmark_file"])
    config = DatasetBuildConfig(
        industry_code=str(dataset["industry_code"]),
        start_date=str(dataset["start_date"]),
        end_date=str(dataset["end_date"]),
        factor_lookback=int(dataset["factor_lookback"]),
        factor_skip_recent=int(dataset["factor_skip_recent"]),
        mad_multiplier=float(dataset["mad_multiplier"]),
        membership_mode=str(dataset["membership_mode"]),
    )
    samples, audit = build_monthly_momentum_dataset(stocks, benchmark, members, config)
    output_path = PROJECT_ROOT / dataset["output_file"]
    audit_path = PROJECT_ROOT / dataset["audit_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(samples)} monthly stock samples to {output_path}")
    print(f"Saved {len(audit)} monthly audit rows to {audit_path}")


if __name__ == "__main__":
    main()
