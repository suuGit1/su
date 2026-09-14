"""Synthetic Power IoT-enabled VPP environment for DT-C3 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
import math
import time

import numpy as np

from .communication import CommunicationChannel, CommunicationConfig
from .digital_twin import DigitalTwinSynchronizer, TwinConfig
from .grid import IEEE33BusInterface
from .safety import SafetyLimits, SafetyShield, SafetyReport
from .ccc import C3ResourceManager, C3Config


@dataclass
class VPPEnvConfig:
    horizon_steps: int = 96
    dt_hours: float = 0.25
    seed: int = 42
    prediction_error: float = 0.0
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
    use_grid_constraints: bool = True
    extreme_event: str | None = None
    computation_latency_budget_s: float = 0.05
    computation_cost_per_s: float = 0.02
    use_digital_twin: bool = True
    dt_preview_reward_weight: float = 4.0
    c3_config: C3Config = field(default_factory=C3Config)


@dataclass
class StepResult:
    state: dict[str, float]
    reward: float
    done: bool
    info: dict[str, Any]


class PowerIoTVPPEnvironment:
    """A compact VPP simulator with non-ideal IoT measurements and DT support."""

    def __init__(self, config: VPPEnvConfig | None = None) -> None:
        self.config = config or VPPEnvConfig()
        self.config.safety_limits = replace(self.config.safety_limits, dt_hours=self.config.dt_hours)
        self.rng = np.random.default_rng(self.config.seed)
        self.channel = CommunicationChannel(self.config.communication)
        self.twin = DigitalTwinSynchronizer(TwinConfig(
            dt_hours=self.config.dt_hours,
            charge_efficiency=self.config.safety_limits.charge_efficiency,
            discharge_efficiency=self.config.safety_limits.discharge_efficiency,
        ))
        self.shield = SafetyShield(self.config.safety_limits)
        self.grid = IEEE33BusInterface()
        self.c3 = C3ResourceManager(self.config.c3_config)
        self.t = 0
        self.profiles = self._make_profiles()
        self.true_state: dict[str, float] = {}
        self.last_info: dict[str, Any] = {}
        self.reset()

    def reset(self) -> dict[str, float]:
        self.t = 0
        self.channel.reset()
        self.c3.reset()
        self.twin.reset()
        self.true_state = {}
        self.last_info = {}
        self._observation_cache = {}
        self._observation_step = None
        self._billed_communication_cost = 0.0
        self.true_state = self._true_state_at(0)
        self.twin.seed_state("ess", {"soc": self.true_state["ess_soc"], "capacity_kwh": self.config.safety_limits.ess_capacity_kwh}, 0)
        self.twin.seed_state("ev", {"soc": self.true_state["ev_soc"], "capacity_kwh": self.config.safety_limits.ev_capacity_kwh}, 0)
        self.twin.seed_state("load", {"power_kw": self.true_state["load_kw"], "price": self.true_state["price"]}, 0)
        self.twin.seed_state("pv", {"power_kw": self.true_state["pv_kw"]}, 0)
        self.twin.seed_state("wind", {"power_kw": self.true_state["wind_kw"]}, 0)
        return self.observe(use_digital_twin=self.config.use_digital_twin)

    def observe(self, use_digital_twin: bool = True) -> dict[str, float]:
        if self._observation_step == self.t:
            return dict(self._observation_cache[use_digital_twin])
        device_states = {
            "ess": {"soc": self.true_state["ess_soc"], "capacity_kwh": self.config.safety_limits.ess_capacity_kwh},
            "ev": {"soc": self.true_state["ev_soc"], "capacity_kwh": self.config.safety_limits.ev_capacity_kwh},
            "load": {"power_kw": self.true_state["load_kw"], "price": self.true_state["price"]},
            "pv": {"power_kw": self.true_state["pv_kw"]},
            "wind": {"power_kw": self.true_state["wind_kw"]},
        }
        tx = self.channel.transmit_batch(device_states, self.t)
        observations = {k: v.observation for k, v in tx.items()}
        comms = {k: v.communication for k, v in tx.items()}
        if self.config.use_digital_twin:
            self.twin.update_batch(observations, comms, true_states=device_states)
        self._observation_cache = {
            True: self.twin.aggregate_state(),
            False: self._aggregate_raw_observations(observations),
        }
        for obs in self._observation_cache.values():
            obs["avg_aoi"] = float(sum(c.aoi for c in comms.values()) / len(comms))
            obs["packet_loss_ratio"] = self.channel.loss_ratio()
            obs["step"] = float(self.t)
        self._observation_step = self.t
        return dict(self._observation_cache[use_digital_twin])

    def step(self, action: dict[str, float], apply_safety: bool = True, use_grid: bool | None = None) -> StepResult:
        if self.t >= self.config.horizon_steps:
            raise RuntimeError("回合已结束，请先 reset")
        if not all(math.isfinite(float(v)) for v in action.values()):
            raise ValueError("动作必须为有限数值")
        use_grid = self.config.use_grid_constraints if use_grid is None else use_grid
        t0 = time.perf_counter()
        observed_state = self.observe(use_digital_twin=self.config.use_digital_twin)
        dt_preview = self.twin.preview_action_safety(action, self.config.safety_limits) if self.config.use_digital_twin else {"safe": True, "risks": []}
        c3_report = self.c3.evaluate(action, observed_state, dt_preview)
        self.channel.apply_c3_report(c3_report)
        safety_report: SafetyReport | None = self.shield.project(observed_state, action) if apply_safety else None
        applied = safety_report.corrected_action if safety_report else {k: float(v) for k, v in action.items()}

        ess_soc = self._next_soc(self.true_state["ess_soc"], applied.get("ess_power_kw", 0.0), self.config.safety_limits.ess_capacity_kwh)
        ev_soc = self._next_soc(self.true_state["ev_soc"], applied.get("ev_power_kw", 0.0), self.config.safety_limits.ev_capacity_kwh)
        # 并网模式下主网承担实际功率差额，计划功率仅用于偏差记录。
        renewable_kw = self.true_state["pv_kw"] + self.true_state["wind_kw"]
        requested_curtailment = float(applied.get("curtailment_kw", 0.0))
        curtailment_kw = min(renewable_kw, max(0.0, requested_curtailment))
        applied["curtailment_kw"] = curtailment_kw
        grid_power_kw = (self.true_state["load_kw"] - renewable_kw + curtailment_kw
                         - applied.get("ess_power_kw", 0.0) - applied.get("ev_power_kw", 0.0)
                         - applied.get("dr_kw", 0.0))
        planned_grid_power_kw = applied.get("grid_power_kw", grid_power_kw)
        applied["grid_power_kw"] = grid_power_kw
        post_state = {**self.true_state, "ess_soc": ess_soc, "ev_soc": ev_soc}
        physical_violations = [v.name for v in self.shield.check_state(post_state)]
        limits = self.config.safety_limits
        checks = {
            "ess_power": abs(applied.get("ess_power_kw", 0.0)) > limits.ess_power_max_kw + 1e-9,
            "ev_power": abs(applied.get("ev_power_kw", 0.0)) > limits.ev_power_max_kw + 1e-9,
            "dr_power": not -1e-9 <= applied.get("dr_kw", 0.0) <= min(limits.dr_max_kw, self.true_state["load_kw"]) + 1e-9,
            "grid_import": grid_power_kw > limits.grid_import_max_kw + 1e-9,
            "grid_export": grid_power_kw < -limits.grid_export_max_kw - 1e-9,
        }
        physical_violations.extend(name for name, violated in checks.items() if violated)
        cumulative_communication_cost = self.channel.total_communication_cost()
        communication_cost = cumulative_communication_cost - self._billed_communication_cost
        self._billed_communication_cost = cumulative_communication_cost
        preliminary_decision_time = time.perf_counter() - t0
        computation_cost = preliminary_decision_time * self.config.computation_cost_per_s
        latency_violation = 1 if preliminary_decision_time > self.config.computation_latency_budget_s else 0
        cost = self._cost(grid_power_kw, applied, curtailment_kw) + communication_cost + computation_cost + c3_report.c3_total_penalty
        dt_risk_penalty = 0.0 if dt_preview.get("safe", True) else self.config.dt_preview_reward_weight * len(dt_preview.get("risks", []))
        reward = -cost - 2.0 * latency_violation - dt_risk_penalty
        violations = len(physical_violations)
        reward -= 10.0 * violations
        grid_violations: list[str] = []
        if use_grid:
            injections = self.grid.vpp_to_bus_injections(self.true_state, applied)
            pf = self.grid.run_power_flow(injections)
            grid_violations = pf.violations
            violations += len(grid_violations)
            reward -= 10.0 * len(grid_violations)

        self.t += 1
        if self.t < self.config.horizon_steps:
            self.true_state = self._true_state_at(self.t)
        self.true_state["ess_soc"] = ess_soc
        self.true_state["ev_soc"] = ev_soc
        done = self.t >= self.config.horizon_steps
        decision_time = time.perf_counter() - t0
        comm_stats = self.channel.stats()
        info = {
            "applied_action": applied,
            "safety_report": None if safety_report is None else safety_report.as_dict(),
            "cost": cost,
            "curtailment_kw": curtailment_kw,
            "grid_power_kw": grid_power_kw,
            "constraint_violations": violations,
            "physical_violations": physical_violations,
            "post_state": post_state,
            "planned_grid_power_kw": planned_grid_power_kw,
            "grid_deviation_kw": grid_power_kw - planned_grid_power_kw,
            "pre_shield_violations": 0 if safety_report is None else safety_report.violation_count,
            "grid_violations": grid_violations,
            "decision_time_s": decision_time,
            "computation_cost": computation_cost,
            "latency_violation": latency_violation,
            "dt_error": self.twin.mean_reconstruction_error(),
            "avg_aoi": self.channel.average_aoi(),
            "avg_delay_steps": comm_stats["avg_delay_steps"],
            "delivery_ratio": comm_stats["delivery_ratio"],
            "communication_overhead_kb": comm_stats["communication_overhead_kb"],
            "communication_cost": communication_cost,
            "dt_preview_safe": bool(dt_preview.get("safe", True)),
            "dt_preview_risk_count": int(len(dt_preview.get("risks", []))),
            "dt_preview_penalty": dt_risk_penalty,
            **c3_report.as_dict(),
            **self.channel.resource_state(),
        }
        self.last_info = info
        return StepResult(state=self.observe(use_digital_twin=self.config.use_digital_twin) if not done else {}, reward=reward, done=done, info=info)

    def _next_soc(self, soc: float, power_kw: float, capacity_kwh: float) -> float:
        if power_kw >= 0:
            soc -= power_kw * self.config.dt_hours / (capacity_kwh * self.config.safety_limits.discharge_efficiency)
        else:
            soc += abs(power_kw) * self.config.dt_hours * self.config.safety_limits.charge_efficiency / capacity_kwh
        # 保留不可行状态，避免裁剪掩盖越限；这里只提供仿真诊断。
        return soc

    def _cost(self, grid_power_kw: float, action: dict[str, float], curtailment_kw: float) -> float:
        energy = grid_power_kw * self.config.dt_hours
        price = self.true_state["price"]
        buy_cost = max(0.0, energy) * price
        sell_revenue = max(0.0, -energy) * price * 0.65
        degradation = 0.01 * (abs(action.get("ess_power_kw", 0.0)) + abs(action.get("ev_power_kw", 0.0))) * self.config.dt_hours
        dr_cost = 0.03 * action.get("dr_kw", 0.0) * self.config.dt_hours
        curtailment_cost = 0.05 * curtailment_kw * self.config.dt_hours
        carbon_cost = 0.02 * max(0.0, energy)
        return buy_cost - sell_revenue + degradation + dr_cost + curtailment_cost + carbon_cost

    def _aggregate_raw_observations(self, observations: dict[str, dict[str, Any] | None]) -> dict[str, float]:
        def val(dev: str, key: str, default: float) -> float:
            obs = observations.get(dev)
            if obs is None:
                return default
            return float(obs.get(key, default))

        return {
            "pv_kw": val("pv", "power_kw", 0.0),
            "wind_kw": val("wind", "power_kw", 0.0),
            "load_kw": val("load", "power_kw", 0.0),
            "ess_soc": val("ess", "soc", 0.5),
            "ev_soc": val("ev", "soc", 0.5),
            "price": val("load", "price", 0.15),
        }

    def _make_profiles(self) -> dict[str, np.ndarray]:
        n = self.config.horizon_steps
        x = np.arange(n)
        day = 2 * np.pi * x / max(1, n)
        load = 650 + 180 * np.sin(day - 0.7) + 80 * np.sin(2 * day) + self.rng.normal(0, 15, n)
        pv = 420 * np.maximum(0, np.sin(day - 0.2)) ** 1.7
        wind = 140 + 50 * np.sin(day + 1.2) + self.rng.normal(0, 10, n)
        price = 0.12 + 0.08 * (load - load.min()) / max(1e-6, load.max() - load.min())
        if self.config.extreme_event == "pv_drop":
            pv[n // 2 : n // 2 + max(2, n // 8)] *= 0.25
        elif self.config.extreme_event == "load_spike":
            load[n // 2 : n // 2 + max(2, n // 8)] *= 1.35
        elif self.config.extreme_event == "price_spike":
            price[n // 2 : n // 2 + max(2, n // 8)] *= 2.0
        if self.config.prediction_error > 0:
            scale = self.rng.normal(1.0, self.config.prediction_error, (3, n))
            pv *= np.clip(scale[0], 0.0, 2.0)
            wind *= np.clip(scale[1], 0.0, 2.0)
            load *= np.clip(scale[2], 0.2, 2.0)
        return {"load_kw": np.maximum(load, 50), "pv_kw": np.maximum(pv, 0), "wind_kw": np.maximum(wind, 0), "price": price}

    def _true_state_at(self, step: int) -> dict[str, float]:
        idx = min(step, self.config.horizon_steps - 1)
        base_ev = 0.55 + 0.1 * math.sin(2 * math.pi * idx / max(1, self.config.horizon_steps))
        if self.true_state:
            base_ev = self.true_state.get("ev_soc", base_ev)
        return {
            "load_kw": float(self.profiles["load_kw"][idx]),
            "pv_kw": float(self.profiles["pv_kw"][idx]),
            "wind_kw": float(self.profiles["wind_kw"][idx]),
            "price": float(self.profiles["price"][idx]),
            "ess_soc": float(self.true_state.get("ess_soc", 0.55)) if self.true_state else 0.55,
            "ev_soc": float(base_ev),
        }
