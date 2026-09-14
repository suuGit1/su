"""Advanced paper-level DT-C3 experiments.

This suite is intended for SCI-paper experiments. It trains neural Actor-Critic
baselines, evaluates traditional optimization baselines, runs ablations and
exports convergence/loss/DT-benchmark logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import csv
import json

import numpy as np

from .algorithms import DTC3SafeMARL, NoCommunicationAwareEMS, NoSafetyShieldEMS, RuleBasedEMS
from .communication import CommunicationChannel, CommunicationConfig
from .digital_twin import DigitalTwinBenchmark
from .environment import PowerIoTVPPEnvironment, VPPEnvConfig
from .optimization_baselines import ADMMDERCoordinator, MILPEnergyManagement, MPCEnergyManagement
from .rl import DTC3MAPPOEMS, IACEMS, MADDPGEMS, MAPPOEMS, NoAoIMAPPOEMS, PPOEMS, RLTrainConfig, NeuralEMSBase, describe_torch_device, resolve_torch_device
from .safety import SafetyLimits


@dataclass
class AdvancedExperimentConfig:
    horizon_steps: int = 96
    train_episodes: int = 100
    seeds: tuple[int, ...] = (1, 2, 3)
    output_dir: str = "output/dt_c3_advanced"
    grid_enabled: bool = False
    hidden_dim: int = 32
    update_epochs: int = 2
    include_maddpg: bool = True
    include_optimization: bool = True
    train_neural: bool = True
    device: str = "cpu"


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
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            groups.setdefault((str(row.get("scenario")), str(row.get("method"))), []).append(row)
        for (scenario, method), rows in groups.items():
            out: dict[str, Any] = {"scenario": scenario, "method": method, "n_seeds": len(rows)}
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


class AdvancedDTC3ExperimentSuite:
    """Train/evaluate proposed and baseline algorithms under paper scenarios."""

    def __init__(self, config: AdvancedExperimentConfig | None = None) -> None:
        self.config = config or AdvancedExperimentConfig()
        self.config.device = resolve_torch_device(self.config.device)
        self.result = AdvancedExperimentResult()

    def make_env(
        self,
        seed: int,
        delay_steps: int = 0,
        packet_loss: float = 0.0,
        prediction_error: float = 0.0,
        extreme_event: str | None = None,
        use_digital_twin: bool = True,
    ) -> PowerIoTVPPEnvironment:
        communication = CommunicationConfig(
            max_delay_steps=delay_steps,
            fixed_delay_steps=delay_steps,
            packet_loss_rate=packet_loss,
            bandwidth_kbps=256.0 if delay_steps >= 3 or packet_loss >= 0.2 else 1024.0,
            packet_size_kb=2.5,
            queue_delay_mean_steps=0.30 if delay_steps >= 3 else 0.05,
            command_downlink_delay_steps=1 if delay_steps >= 3 else 0,
            seed=seed,
        )
        return PowerIoTVPPEnvironment(
            VPPEnvConfig(
                horizon_steps=self.config.horizon_steps,
                seed=seed,
                communication=communication,
                prediction_error=prediction_error,
                use_grid_constraints=self.config.grid_enabled,
                safety_limits=SafetyLimits(),
                extreme_event=extreme_event,
                use_digital_twin=use_digital_twin,
            )
        )

    def neural_methods(self, seed: int, safety: bool = True, communication_aware: bool = True, use_aoi: bool = True) -> list[NeuralEMSBase]:
        cfg = RLTrainConfig(
            episodes=self.config.train_episodes,
            seed=seed,
            hidden_dim=self.config.hidden_dim,
            update_epochs=self.config.update_epochs,
            batch_size=128,
            device=self.config.device,
            safety_enabled=safety,
            communication_aware=communication_aware,
            use_aoi=use_aoi,
        )
        methods: list[NeuralEMSBase] = [
            DTC3MAPPOEMS(cfg),
            MAPPOEMS(cfg),
            PPOEMS(cfg),
            IACEMS(cfg),
        ]
        if self.config.include_maddpg:
            methods.append(MADDPGEMS(cfg))
        return methods

    def heuristic_methods(self) -> list[Any]:
        return [DTC3SafeMARL(), RuleBasedEMS(), NoCommunicationAwareEMS(), NoSafetyShieldEMS()]

    def optimization_methods(self) -> list[Any]:
        return [MILPEnergyManagement(), MPCEnergyManagement(), ADMMDERCoordinator()] if self.config.include_optimization else []

    def _train_eval_neural(self, scenario: str, params: dict[str, Any], seed: int, methods: list[NeuralEMSBase] | None = None) -> None:
        if not self.config.train_neural:
            return
        for method in methods or self.neural_methods(seed):
            trace = method.train(lambda s=seed: self.make_env(s, **params), episodes=self.config.train_episodes)
            self.result.add_trace(scenario, trace)
            metrics = method.evaluate(self.make_env(seed + 10_000, **params))
            metrics.update({"scenario": scenario, "seed": seed, "device": self.config.device, **params})
            self.result.add_metrics(metrics)

    def _run_scenario(self, scenario: str, params: dict[str, Any]) -> None:
        for seed in self.config.seeds:
            self._train_eval_neural(scenario, params, seed)
            for method in self.heuristic_methods() + self.optimization_methods():
                metrics = method.evaluate(self.make_env(seed + 20_000, **params))
                metrics.update({"scenario": scenario, "seed": seed, "device": self.config.device, **params})
                self.result.add_metrics(metrics)

    def run_core(self) -> AdvancedExperimentResult:
        self.result = AdvancedExperimentResult()
        self._run_scenario("normal_communication", {"delay_steps": 0, "packet_loss": 0.0})
        self._run_scenario("high_delay", {"delay_steps": 5, "packet_loss": 0.05})
        self._run_scenario("high_packet_loss", {"delay_steps": 2, "packet_loss": 0.30})
        self._run_scenario("prediction_error", {"delay_steps": 1, "packet_loss": 0.05, "prediction_error": 0.20})
        self._run_scenario("extreme_load_spike", {"delay_steps": 2, "packet_loss": 0.10, "extreme_event": "load_spike"})
        self.run_ablation()
        self.run_digital_twin_benchmark()
        return self.result

    def run_ablation(self) -> None:
        params = {"delay_steps": 5, "packet_loss": 0.30, "prediction_error": 0.20}
        for seed in self.config.seeds:
            variants: list[tuple[str, Callable[[], NeuralEMSBase], dict[str, Any]]] = [
                ("full_dt_c3_safemarl", lambda s=seed: DTC3MAPPOEMS(RLTrainConfig(episodes=self.config.train_episodes, seed=s, hidden_dim=self.config.hidden_dim, update_epochs=self.config.update_epochs, device=self.config.device, safety_enabled=True, communication_aware=True, use_aoi=True)), {"use_digital_twin": True}),
                ("without_digital_twin", lambda s=seed: DTC3MAPPOEMS(RLTrainConfig(episodes=self.config.train_episodes, seed=s, hidden_dim=self.config.hidden_dim, update_epochs=self.config.update_epochs, device=self.config.device, safety_enabled=True, communication_aware=True, use_aoi=True)), {"use_digital_twin": False}),
                ("without_communication_awareness", lambda s=seed: DTC3MAPPOEMS(RLTrainConfig(episodes=self.config.train_episodes, seed=s, hidden_dim=self.config.hidden_dim, update_epochs=self.config.update_epochs, device=self.config.device, safety_enabled=True, communication_aware=False, use_aoi=True)), {"use_digital_twin": True}),
                ("without_safety_shield", lambda s=seed: DTC3MAPPOEMS(RLTrainConfig(episodes=self.config.train_episodes, seed=s, hidden_dim=self.config.hidden_dim, update_epochs=self.config.update_epochs, device=self.config.device, safety_enabled=False, communication_aware=True, use_aoi=True)), {"use_digital_twin": True}),
                ("without_aoi", lambda s=seed: NoAoIMAPPOEMS(RLTrainConfig(episodes=self.config.train_episodes, seed=s, hidden_dim=self.config.hidden_dim, update_epochs=self.config.update_epochs, device=self.config.device, safety_enabled=True, communication_aware=True, use_aoi=False)), {"use_digital_twin": True}),
            ]
            for variant, factory, extra_env in variants:
                method = factory()
                p = {**params, **extra_env}
                trace = method.train(lambda s=seed: self.make_env(s, **p), episodes=self.config.train_episodes)
                self.result.add_trace("ablation", trace)
                metrics = method.evaluate(self.make_env(seed + 30_000, **p))
                metrics.update({"scenario": "ablation", "variant": variant, "seed": seed, "device": self.config.device, **p})
                self.result.add_metrics(metrics)

    def run_digital_twin_benchmark(self) -> None:
        scenarios = [
            ("dt_normal", 0, 0.0),
            ("dt_high_delay", 5, 0.05),
            ("dt_high_loss", 2, 0.30),
        ]
        for name, delay, loss in scenarios:
            for seed in self.config.seeds:
                env = self.make_env(seed, delay_steps=delay, packet_loss=loss)
                channel = CommunicationChannel(env.config.communication)
                true_series: list[dict[str, float]] = []
                observed_series: list[dict[str, float] | None] = []
                for t in range(self.config.horizon_steps):
                    st = env._true_state_at(t)  # research benchmark helper
                    flat = {
                        "load_kw": float(st["load_kw"]),
                        "pv_kw": float(st["pv_kw"]),
                        "wind_kw": float(st["wind_kw"]),
                        "ess_soc": float(st["ess_soc"]),
                        "ev_soc": float(st["ev_soc"]),
                    }
                    res = channel.transmit("vpp", flat, t)
                    true_series.append(flat)
                    observed_series.append(res.observation)
                for metric in DigitalTwinBenchmark().run(true_series, observed_series):
                    row = {"scenario": name, "seed": seed, "estimator": metric.name, "overall_rmse": metric.overall_rmse}
                    for k, v in metric.rmse.items():
                        row[f"{k}_rmse"] = v
                    self.result.dt_benchmark_rows.append(row)

    def save(self) -> dict[str, Path]:
        return self.result.save(self.config.output_dir)
