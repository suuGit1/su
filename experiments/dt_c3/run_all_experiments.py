"""Run all DT-C3 paper experiments.

Usage:
    python experiments/dt_c3/run_all_experiments.py --horizon 96 --output output/dt_c3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vpp.dt_c3 import DTC3ExperimentSuite, ExperimentSuiteConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=96, help="Number of dispatch steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/dt_c3")
    parser.add_argument("--no-grid", action="store_true", help="Disable simplified IEEE 33-bus checks")
    args = parser.parse_args()

    suite = DTC3ExperimentSuite(
        ExperimentSuiteConfig(
            horizon_steps=args.horizon,
            seed=args.seed,
            output_dir=args.output,
            grid_enabled=not args.no_grid,
        )
    )
    result = suite.run_all()
    csv_path, json_path = suite.save()
    print(f"Completed {len(result.rows)} experiment rows")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
