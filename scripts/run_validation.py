"""Run momentum lookback and recent-period-skip sensitivity checks."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtester.analytics.validation import parameter_sensitivity
from backtester.config import load_config_directory
from run_backtest import run_research


def main() -> None:
    parser = argparse.ArgumentParser(description="Run momentum parameter sensitivity")
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    args = parser.parse_args()
    config = load_config_directory(args.config_dir)
    factor_config = config["factor"]

    lookback_table = parameter_sensitivity(
        factor_config["robustness_lookbacks"],
        lambda lookback: run_research(
            args.config_dir,
            lookback_override=int(lookback),
            save_outputs=False,
        ).metrics,
        "lookback",
    )
    skip_table = parameter_sensitivity(
        factor_config["robustness_skip_periods"],
        lambda skip: run_research(
            args.config_dir,
            skip_override=int(skip),
            save_outputs=False,
        ).metrics,
        "skip_recent",
    )

    output = PROJECT_ROOT / "data" / "results"
    output.mkdir(parents=True, exist_ok=True)
    lookback_table.to_csv(output / "lookback_sensitivity.csv")
    skip_table.to_csv(output / "skip_sensitivity.csv")
    print("Lookback sensitivity")
    print(lookback_table.to_string())
    print("\nSkip-period sensitivity")
    print(skip_table.to_string())


if __name__ == "__main__":
    main()

