"""DT-C3 Safe-MARL research prototype.

This file provides a paper-ready algorithm scaffold with a deterministic,
communication-aware policy that is fast enough for reproducible experiments.
It is intentionally implemented without PyTorch so the uploaded project remains
installable with the original lightweight dependencies.  The class boundaries
(`act`, `train`, `evaluate`) are compatible with replacing the heuristic actor
by MAPPO/MADDPG later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import time

import numpy as np

from .environment import PowerIoTVPPEnvironment
from .safety import SafetyShield


class EnergyManagementPolicy(Protocol):
    name: str

    def act(self, state: dict[str, float]) -> dict[str, float]:
        ...


@dataclass
class DTC3SafeMARLConfig:
    ess_power_max_kw: float = 250.0
    ev_power_max_kw: float = 300.0
    dr_max_kw: float = 150.0
    conservative_aoi_threshold: float = 3.0
    high_price: float = 0.17
    low_price: float = 0.13
    renewable_surplus_margin_kw: float = 50.0
    safety_enabled: bool = True
    communication_aware: bool = True
    seed: int = 42


@dataclass
class TrainingTrace:
    rewards: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    violations: list[int] = field(default_factory=list)


class DTC3SafeMARL:
    """Communication-aware, safety-shielded multi-agent EMS prototype."""

    name = "dt_c3_safemarl"

    def __init__(self, config: DTC3SafeMARLConfig | None = None, shield: SafetyShield | None = None) -> None:
        self.config = config or DTC3SafeMARLConfig()
        self.shield = shield or SafetyShield()
        self.rng = np.random.default_rng(self.config.seed)
        self.training_trace = TrainingTrace()

    def act(self, state: dict[str, float]) -> dict[str, float]:
        load = float(state.get("load_kw", 0.0))
        pv = float(state.get("pv_kw", 0.0))
        wind = float(state.get("wind_kw", 0.0))
        price = float(state.get("price", 0.15))
        ess_soc = float(state.get("ess_soc", 0.5))
        ev_soc = float(state.get("ev_soc", 0.5))
        aoi = float(state.get("avg_aoi", 0.0))
        renewable = pv + wind
        net_load = load - renewable

        # Conservative operation when IoT information is stale.
        confidence_scale = 1.0
        if self.config.communication_aware and aoi >= self.config.conservative_aoi_threshold:
            confidence_scale = max(0.35, 1.0 - 0.12 * (aoi - self.config.conservative_aoi_threshold + 1))

        ess_power = 0.0
        ev_power = 0.0
        dr_kw = 0.0

        if renewable - load > self.config.renewable_surplus_margin_kw or price <= self.config.low_price:
            # Charge storage/EVs using cheap or surplus renewable energy.
            if ess_soc < 0.85:
                ess_power = -min(self.config.ess_power_max_kw, max(0.0, renewable - load) * 0.6 + 60)
            if ev_soc < 0.85:
                ev_power = -min(self.config.ev_power_max_kw, max(0.0, renewable - load) * 0.4 + 40)
        elif price >= self.config.high_price or net_load > 0:
            # Discharge during high-price / high-net-load periods.
            if ess_soc > 0.25:
                ess_power = min(self.config.ess_power_max_kw, max(0.0, net_load) * 0.45 + 40)
            if ev_soc > 0.35 and price >= self.config.high_price:
                ev_power = min(self.config.ev_power_max_kw, max(0.0, net_load) * 0.25 + 30)
            if net_load > 900:
                dr_kw = min(self.config.dr_max_kw, (net_load - 900) * 0.4)

        action = {
            "ess_power_kw": ess_power * confidence_scale,
            "ev_power_kw": ev_power * confidence_scale,
            "dr_kw": dr_kw,
        }
        if self.config.safety_enabled:
            return self.shield.project(state, action).corrected_action
        return action

    def train(self, env_factory: callable, episodes: int = 5) -> TrainingTrace:
        """Run lightweight training/evaluation episodes.

        The current actor is heuristic, so this method collects learning-like
        traces and can be replaced by MAPPO updates later without changing the
        experiment suite.
        """
        self.training_trace = TrainingTrace()
        for _ in range(episodes):
            env = env_factory()
            state = env.reset()
            done = False
            total_reward = 0.0
            total_cost = 0.0
            total_violations = 0
            while not done:
                action = self.act(state)
                result = env.step(action, apply_safety=self.config.safety_enabled)
                state = result.state
                done = result.done
                total_reward += result.reward
                total_cost += float(result.info.get("cost", 0.0))
                total_violations += int(result.info.get("constraint_violations", 0))
            self.training_trace.rewards.append(total_reward)
            self.training_trace.costs.append(total_cost)
            self.training_trace.violations.append(total_violations)
        return self.training_trace

    def evaluate(self, env: PowerIoTVPPEnvironment) -> dict[str, Any]:
        state = env.reset()
        done = False
        total_reward = 0.0
        total_cost = 0.0
        total_violations = 0
        total_curtailment = 0.0
        decision_times: list[float] = []
        grid_power: list[float] = []
        soc: list[float] = []
        while not done:
            t0 = time.perf_counter()
            action = self.act(state)
            result = env.step(action, apply_safety=self.config.safety_enabled)
            decision_times.append(time.perf_counter() - t0 + float(result.info.get("decision_time_s", 0.0)))
            state = result.state
            done = result.done
            total_reward += result.reward
            total_cost += float(result.info.get("cost", 0.0))
            total_violations += int(result.info.get("constraint_violations", 0))
            total_curtailment += float(result.info.get("curtailment_kw", 0.0)) * env.config.dt_hours
            grid_power.append(float(result.info.get("grid_power_kw", 0.0)))
            if state:
                soc.append(float(state.get("ess_soc", 0.5)))
        return {
            "method": self.name,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "constraint_violations": total_violations,
            "curtailment_kwh": total_curtailment,
            "avg_aoi": env.channel.average_aoi(),
            "dt_reconstruction_error": env.twin.mean_reconstruction_error(),
            "avg_decision_time_s": float(np.mean(decision_times)) if decision_times else 0.0,
            "grid_power_std_kw": float(np.std(grid_power)) if grid_power else 0.0,
            "soc_min": float(np.min(soc)) if soc else 0.0,
            "soc_max": float(np.max(soc)) if soc else 0.0,
        }


class RuleBasedEMS(DTC3SafeMARL):
    name = "rule_based_ems"

    def __init__(self) -> None:
        super().__init__(DTC3SafeMARLConfig(safety_enabled=True, communication_aware=False))

    def act(self, state: dict[str, float]) -> dict[str, float]:
        net_load = float(state.get("load_kw", 0.0)) - float(state.get("pv_kw", 0.0)) - float(state.get("wind_kw", 0.0))
        ess_soc = float(state.get("ess_soc", 0.5))
        power = 0.0
        if net_load > 300 and ess_soc > 0.25:
            power = min(180.0, net_load * 0.35)
        elif net_load < -50 and ess_soc < 0.85:
            power = -min(180.0, abs(net_load) * 0.35)
        return self.shield.project(state, {"ess_power_kw": power, "ev_power_kw": 0.0, "dr_kw": 0.0}).corrected_action


class NoDigitalTwinEMS(DTC3SafeMARL):
    name = "without_digital_twin"


class NoCommunicationAwareEMS(DTC3SafeMARL):
    name = "without_communication_awareness"

    def __init__(self) -> None:
        super().__init__(DTC3SafeMARLConfig(communication_aware=False, safety_enabled=True))


class NoSafetyShieldEMS(DTC3SafeMARL):
    name = "without_safety_shield"

    def __init__(self) -> None:
        super().__init__(DTC3SafeMARLConfig(communication_aware=True, safety_enabled=False))


class RandomEMS(DTC3SafeMARL):
    name = "random_ems"

    def __init__(self, seed: int = 7) -> None:
        super().__init__(DTC3SafeMARLConfig(safety_enabled=True, communication_aware=False, seed=seed))

    def act(self, state: dict[str, float]) -> dict[str, float]:
        action = {
            "ess_power_kw": float(self.rng.uniform(-180, 180)),
            "ev_power_kw": float(self.rng.uniform(-120, 120)),
            "dr_kw": float(self.rng.uniform(0, 80)),
        }
        return self.shield.project(state, action).corrected_action

# Neural Actor-Critic / MAPPO / MADDPG additions.  Kept at the bottom so the
# previous lightweight heuristic API remains backward compatible.
try:  # pragma: no cover - import availability depends on optional torch
    from .rl import (
        RLTrainConfig,
        RLTrainingTrace,
        PPOEMS,
        IACEMS,
        MAPPOEMS,
        MADDPGEMS,
        DTC3MAPPOEMS,
        run_multi_seed_training,
        torch_available,
    )
except Exception:  # pragma: no cover
    RLTrainConfig = None  # type: ignore
    RLTrainingTrace = None  # type: ignore
    PPOEMS = None  # type: ignore
    IACEMS = None  # type: ignore
    MAPPOEMS = None  # type: ignore
    MADDPGEMS = None  # type: ignore
    DTC3MAPPOEMS = None  # type: ignore
    run_multi_seed_training = None  # type: ignore
    def torch_available() -> bool:  # type: ignore
        return False
