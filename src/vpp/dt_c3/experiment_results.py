"""实验结果汇总与导出，不依赖强化学习训练库。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import json
import numpy as np

@dataclass
class AdvancedExperimentResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    convergence_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    dt_benchmark_rows: list[dict[str, Any]] = field(default_factory=list)
    training_log_lines: list[str] = field(default_factory=list)

    def add_metrics(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def add_trace(self, scenario: str, trace: Any) -> None:
        n = max(len(trace.episode_rewards), len(trace.actor_losses), len(trace.critic_losses))
        self.training_log_lines.append(f"[{scenario}] {trace.method} seed={trace.seed} episodes={len(trace.episode_rewards)} last_reward={trace.episode_rewards[-1] if trace.episode_rewards else None} last_actor_loss={trace.actor_losses[-1] if trace.actor_losses else None} last_critic_loss={trace.critic_losses[-1] if trace.critic_losses else None}")
        for i in range(n):
            self.convergence_rows.append({
                "scenario": scenario,
                "method": trace.method,
                "seed": trace.seed,
                "episode": i + 1,
                "episode_reward": trace.episode_rewards[i] if i < len(trace.episode_rewards) else None,
                "episode_cost": trace.episode_costs[i] if i < len(trace.episode_costs) else None,
                "episode_violations": trace.episode_violations[i] if i < len(trace.episode_violations) else None,
                "actor_loss": trace.actor_losses[i] if i < len(trace.actor_losses) else None,
                "critic_loss": trace.critic_losses[i] if i < len(trace.critic_losses) else None,
                "entropy": trace.entropy[i] if i < len(trace.entropy) else None,
            })

    def build_summary(self) -> None:
        self.summary_rows.clear()
        if not self.rows:
            return
        keys = [
            "total_cost", "total_reward", "constraint_violations", "curtailment_kwh",
            "avg_aoi", "avg_delay_steps", "delivery_ratio", "communication_overhead_kb",
            "dt_reconstruction_error", "avg_decision_time_s", "grid_power_std_kw",
            "soc_min", "soc_max", "pre_shield_violations", "safety_correction_kw_mean",
            "actor_loss_last", "critic_loss_last", "train_reward_last", "computation_latency_s",
            "c3_total_penalty", "dt_preview_penalty", "offloading_cost", "computation_queue_delay_s", "deadline_violations", "runtime_bandwidth_kbps", "runtime_upload_interval_steps", "effective_dt_update_rate",
        ]
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            groups.setdefault((str(row.get("scenario")), str(row.get("method")), str(row.get("variant", ""))), []).append(row)
        for (scenario, method, variant), rows in groups.items():
            out: dict[str, Any] = {"scenario": scenario, "method": method, "variant": variant, "n_seeds": len({r.get("seed") for r in rows}), "n_records": len(rows)}
            for k in keys:
                vals = [float(r[k]) for r in rows if k in r and r[k] is not None and isinstance(r[k], (int, float))]
                if vals:
                    out[f"{k}_mean"] = float(np.mean(vals))
                    out[f"{k}_std"] = float(np.std(vals, ddof=0))
            self.summary_rows.append(out)

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = sorted({k for row in rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        self.build_summary()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "raw_csv": out / "advanced_raw_results.csv",
            "summary_csv": out / "advanced_summary_mean_std.csv",
            "convergence_csv": out / "advanced_convergence_losses.csv",
            "dt_benchmark_csv": out / "digital_twin_benchmark_rmse.csv",
            "training_log": out / "training_log.txt",
            "raw_json": out / "advanced_raw_results.json",
        }
        self._write_csv(paths["raw_csv"], self.rows)
        self._write_csv(paths["summary_csv"], self.summary_rows)
        self._write_csv(paths["convergence_csv"], self.convergence_rows)
        self._write_csv(paths["dt_benchmark_csv"], self.dt_benchmark_rows)
        paths["training_log"].write_text("\n".join(self.training_log_lines), encoding="utf-8")
        paths["raw_json"].write_text(json.dumps(self.rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return paths

