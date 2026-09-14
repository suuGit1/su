"""Generate paper-ready DT-C3 figures from advanced experiment CSV outputs.

Usage:
    PYTHONPATH=src python experiments/dt_c3/make_paper_figures.py \
        --input output/dt_c3_advanced --output output/dt_c3_advanced/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="output/dt_c3_advanced")
    parser.add_argument("--output", type=str, default="output/dt_c3_advanced/figures")
    args = parser.parse_args()
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"pandas/matplotlib are required for plotting: {exc}")

    inp = Path(args.input)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(inp / "advanced_summary_mean_std.csv")
    conv = pd.read_csv(inp / "advanced_convergence_losses.csv")
    dt = pd.read_csv(inp / "digital_twin_benchmark_rmse.csv")

    def bar(metric: str, filename: str, scenario: str | None = None) -> None:
        df = summary.copy()
        if scenario:
            df = df[df["scenario"] == scenario]
        col = f"{metric}_mean"
        if col not in df:
            return
        ax = df.plot(kind="bar", x="method", y=col, legend=False, figsize=(10, 4))
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} comparison" + (f" - {scenario}" if scenario else ""))
        plt.tight_layout(); plt.savefig(out / filename, dpi=300); plt.close()

    bar("total_cost", "cost_comparison_normal.png", "normal_communication")
    bar("constraint_violations", "constraint_violations.png")
    bar("avg_aoi", "aoi_comparison.png")
    bar("delivery_ratio", "delivery_ratio.png")
    bar("communication_overhead_kb", "communication_overhead.png")
    bar("c3_total_penalty", "c3_penalty.png")
    bar("computation_latency_s", "computation_latency.png")
    bar("computation_queue_delay_s", "computation_queue_delay.png")
    bar("deadline_violations", "deadline_violations.png")
    bar("runtime_bandwidth_kbps", "runtime_bandwidth.png")

    if not conv.empty:
        for metric, fname in [("episode_reward", "reward_convergence.png"), ("actor_loss", "actor_loss.png"), ("critic_loss", "critic_loss.png")]:
            if metric in conv:
                ax = conv.pivot_table(index="episode", columns="method", values=metric, aggfunc="mean").plot(figsize=(8, 4))
                ax.set_ylabel(metric); ax.set_title(metric.replace("_", " "))
                plt.tight_layout(); plt.savefig(out / fname, dpi=300); plt.close()

    if not dt.empty:
        ax = dt.pivot_table(index="estimator", columns="scenario", values="overall_rmse", aggfunc="mean").plot(kind="bar", figsize=(10, 4))
        ax.set_ylabel("overall RMSE"); ax.set_title("Digital twin estimator RMSE")
        plt.tight_layout(); plt.savefig(out / "digital_twin_rmse.png", dpi=300); plt.close()

    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
