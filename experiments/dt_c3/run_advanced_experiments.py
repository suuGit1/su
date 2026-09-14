"""Run advanced DT-C3 neural Actor-Critic experiments.

Examples
--------
Quick smoke run:
    PYTHONPATH=src python experiments/dt_c3/run_advanced_experiments.py \
        --horizon 8 --episodes 1 --seeds 1 --output output/dt_c3_advanced_smoke --no-maddpg

Paper-scale run:
    PYTHONPATH=src python experiments/dt_c3/run_advanced_experiments.py \
        --horizon 96 --episodes 100 --seeds 1 2 3 4 5 --output output/dt_c3_advanced
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vpp.dt_c3 import AdvancedDTC3ExperimentSuite, AdvancedExperimentConfig  # noqa: E402
from vpp.dt_c3.rl import describe_torch_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=str, default="output/dt_c3_advanced")
    parser.add_argument("--grid", action="store_true", help="Enable simplified IEEE 33-bus checks")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--no-maddpg", action="store_true")
    parser.add_argument("--no-optimization", action="store_true", help="Skip MILP/MPC/ADMM baselines")
    parser.add_argument("--no-neural", action="store_true", help="Skip neural training for fast baseline checks")
    parser.add_argument("--device", type=str, default="auto", help="Torch device for neural RL training: auto, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    print(f"[DT-C3] Requested device: {args.device}")
    print(f"[DT-C3] Using device: {describe_torch_device(args.device)}")

    suite = AdvancedDTC3ExperimentSuite(
        AdvancedExperimentConfig(
            horizon_steps=args.horizon,
            train_episodes=args.episodes,
            seeds=tuple(args.seeds),
            output_dir=args.output,
            grid_enabled=args.grid,
            hidden_dim=args.hidden_dim,
            update_epochs=args.update_epochs,
            include_maddpg=not args.no_maddpg,
            include_optimization=not args.no_optimization,
            train_neural=not args.no_neural,
            device=args.device,
        )
    )
    result = suite.run_core()
    paths = suite.save()
    print(f"Completed {len(result.rows)} advanced experiment rows")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
