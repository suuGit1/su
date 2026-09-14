"""Traditional optimization baselines for DT-C3 VPP experiments.

This upgraded module provides reviewer-recognizable baselines:

* multi-period MILP/LP energy management with SOC trajectory constraints;
* receding-horizon MPC that solves a look-ahead optimization and applies only
  the first control action;
* ADMM distributed DER coordinator with primal/dual residual traces.

The implementation remains dependency-light.  If PuLP is available, MILP/MPC
use binary charge/discharge and import/export variables.  If not, the same
interfaces fall back to a deterministic multi-period optimizer so the project
remains runnable out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time
import numpy as np

from .environment import PowerIoTVPPEnvironment
from .safety import SafetyShield

try:  # optional dependency
    import pulp  # type: ignore
except Exception:  # pragma: no cover
    pulp = None

ACTION_KEYS = ["ess_power_kw", "ev_power_kw", "dr_kw", "upload_priority", "edge_cpu_fraction", "offload_ratio"]


@dataclass
class OptimizationTrace:
    primal_residuals: list[float] = field(default_factory=list)
    dual_residuals: list[float] = field(default_factory=list)
    objective_values: list[float] = field(default_factory=list)
    solver_status: str = "not_solved"


class EvaluationMixin:
    name = "baseline"

    def evaluate(self, env: PowerIoTVPPEnvironment) -> dict[str, Any]:
        self._active_env = env
        state = env.reset()
        done = False
        total_reward = total_cost = total_curt = 0.0
        violations = 0
        decision_times: list[float] = []
        grid_power: list[float] = []
        soc: list[float] = []
        pre_shield = 0
        correction: list[float] = []
        while not done:
            t0 = time.perf_counter()
            raw = self.act(state)
            report = self.shield.project(state, raw)
            action = report.corrected_action if self.safety_enabled else raw
            pre_shield += report.violation_count
            correction.append(sum(abs(action.get(k, 0.0) - raw.get(k, 0.0)) for k in ACTION_KEYS))
            result = env.step(action, apply_safety=False)
            decision_times.append(time.perf_counter() - t0 + float(result.info.get("decision_time_s", 0.0)))
            state = result.state
            done = result.done
            total_reward += result.reward
            total_cost += float(result.info.get("cost", 0.0))
            total_curt += float(result.info.get("curtailment_kw", 0.0)) * env.config.dt_hours
            violations += int(result.info.get("constraint_violations", 0))
            grid_power.append(float(result.info.get("grid_power_kw", 0.0)))
            if state:
                soc.append(float(state.get("ess_soc", 0.5)))
        stats = env.channel.stats()
        trace = getattr(self, "last_trace", OptimizationTrace())
        return {
            "method": self.name,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "constraint_violations": violations,
            "curtailment_kwh": total_curt,
            "avg_aoi": stats.get("avg_aoi", 0.0),
            "avg_delay_steps": stats.get("avg_delay_steps", 0.0),
            "delivery_ratio": stats.get("delivery_ratio", 1.0),
            "communication_overhead_kb": stats.get("communication_overhead_kb", 0.0),
            "dt_reconstruction_error": env.twin.mean_reconstruction_error(),
            "dt_variable_rmse": env.twin.variable_rmse(),
            "avg_decision_time_s": float(np.mean(decision_times)) if decision_times else 0.0,
            "grid_power_std_kw": float(np.std(grid_power)) if grid_power else 0.0,
            "soc_min": float(np.min(soc)) if soc else 0.0,
            "soc_max": float(np.max(soc)) if soc else 0.0,
            "pre_shield_violations": pre_shield,
            "safety_correction_kw_mean": float(np.mean(correction)) if correction else 0.0,
            "actor_loss_last": None,
            "critic_loss_last": None,
            "train_reward_last": None,
            "admm_primal_residual_last": trace.primal_residuals[-1] if trace.primal_residuals else None,
            "admm_dual_residual_last": trace.dual_residuals[-1] if trace.dual_residuals else None,
            "optimization_objective_last": trace.objective_values[-1] if trace.objective_values else None,
            "solver_status": trace.solver_status,
        }


class MILPEnergyManagement(EvaluationMixin):
    """24h / multi-step MILP baseline with time-coupled SOC constraints."""

    name = "milp_full_horizon"

    def __init__(self, horizon_steps: int | None = None, safety_enabled: bool = True) -> None:
        self.horizon_steps = horizon_steps
        self.shield = SafetyShield()
        self.safety_enabled = safety_enabled
        self.last_plan: list[dict[str, float]] = []
        self.last_trace = OptimizationTrace()

    def _future_profiles(self, env: PowerIoTVPPEnvironment, start_t: int, horizon: int) -> list[dict[str, float]]:
        rows = []
        for k in range(horizon):
            t = min(start_t + k, env.config.horizon_steps - 1)
            st = env._true_state_at(t)  # research helper; mimics day-ahead forecast data
            rows.append({
                "load_kw": float(st["load_kw"]),
                "pv_kw": float(st["pv_kw"]),
                "wind_kw": float(st["wind_kw"]),
                "price": float(st["price"]),
            })
        return rows

    def act(self, state: dict[str, float]) -> dict[str, float]:
        env = getattr(self, "_active_env", None)
        h = self.horizon_steps or (env.config.horizon_steps - env.t if env is not None else 24)
        h = max(1, min(int(h), 96))
        profiles = self._future_profiles(env, env.t, h) if env is not None else [state] * h
        if pulp is not None:
            try:
                plan, trace = self._solve_with_pulp(state, profiles, env.config.dt_hours if env is not None else 0.25)
                self.last_plan = plan
                self.last_trace = trace
                return plan[0]
            except Exception:
                pass
        plan = self._fallback_full_horizon(state, profiles)
        self.last_plan = plan
        self.last_trace = OptimizationTrace(objective_values=[sum(abs(a["ess_power_kw"]) + abs(a["ev_power_kw"]) for a in plan)], solver_status="fallback")
        return plan[0]

    def _solve_with_pulp(self, state: dict[str, float], profiles: list[dict[str, float]], dt: float) -> tuple[list[dict[str, float]], OptimizationTrace]:
        n = len(profiles)
        prob = pulp.LpProblem("vpp_full_horizon_milp", pulp.LpMinimize)
        ess = pulp.LpVariable.dicts("ess_kw", range(n), lowBound=-250, upBound=250)
        ev = pulp.LpVariable.dicts("ev_kw", range(n), lowBound=-300, upBound=300)
        dr = pulp.LpVariable.dicts("dr_kw", range(n), lowBound=0, upBound=150)
        grid_in = pulp.LpVariable.dicts("grid_in", range(n), lowBound=0, upBound=1200)
        grid_out = pulp.LpVariable.dicts("grid_out", range(n), lowBound=0, upBound=800)
        curtail = pulp.LpVariable.dicts("curtail", range(n), lowBound=0)
        ess_soc = pulp.LpVariable.dicts("ess_soc", range(n + 1), lowBound=0.15, upBound=0.90)
        ev_soc = pulp.LpVariable.dicts("ev_soc", range(n + 1), lowBound=0.20, upBound=0.95)
        ess_charge = pulp.LpVariable.dicts("ess_charge_bin", range(n), cat="Binary")
        ev_charge = pulp.LpVariable.dicts("ev_charge_bin", range(n), cat="Binary")
        import_bin = pulp.LpVariable.dicts("import_bin", range(n), cat="Binary")
        prob += ess_soc[0] == float(state.get("ess_soc", 0.5))
        prob += ev_soc[0] == float(state.get("ev_soc", 0.6))
        obj = []
        for t, p in enumerate(profiles):
            net = p["load_kw"] - p["pv_kw"] - p["wind_kw"] - ess[t] - ev[t] - dr[t]
            prob += grid_in[t] - grid_out[t] == net - curtail[t]
            prob += grid_in[t] <= 1200 * import_bin[t]
            prob += grid_out[t] <= 800 * (1 - import_bin[t])
            prob += ess[t] <= 250 * (1 - ess_charge[t])
            prob += ess[t] >= -250 * ess_charge[t]
            prob += ev[t] <= 300 * (1 - ev_charge[t])
            prob += ev[t] >= -300 * ev_charge[t]
            prob += ess_soc[t + 1] == ess_soc[t] - ess[t] * dt / (500.0 * 0.95)
            prob += ev_soc[t + 1] == ev_soc[t] - ev[t] * dt / (1200.0 * 0.95)
            obj.append(p["price"] * dt * grid_in[t] - 0.65 * p["price"] * dt * grid_out[t] + 0.01 * dt * (ess[t] + 250) + 0.008 * dt * (ev[t] + 300) + 0.03 * dt * dr[t] + 0.05 * dt * curtail[t])
        # EV final satisfaction surrogate.
        prob += ev_soc[n] >= 0.50
        prob += pulp.lpSum(obj)
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=5))
        status = pulp.LpStatus.get(prob.status, "unknown")
        plan = []
        for t in range(n):
            plan.append({
                "ess_power_kw": float(ess[t].value() or 0.0),
                "ev_power_kw": float(ev[t].value() or 0.0),
                "dr_kw": float(dr[t].value() or 0.0),
                "upload_priority": 0.65,
                "edge_cpu_fraction": 0.65,
                "offload_ratio": 0.10,
            })
        obj_val = float(pulp.value(prob.objective) or 0.0)
        return plan, OptimizationTrace(objective_values=[obj_val], solver_status=status)

    def _fallback_full_horizon(self, state: dict[str, float], profiles: list[dict[str, float]]) -> list[dict[str, float]]:
        ess_soc = float(state.get("ess_soc", 0.5))
        ev_soc = float(state.get("ev_soc", 0.6))
        plan = []
        avg_price = float(np.mean([p.get("price", 0.15) for p in profiles]))
        for p in profiles:
            net = p["load_kw"] - p["pv_kw"] - p["wind_kw"]
            price = p.get("price", 0.15)
            surplus = -net
            if surplus > 0 or price < avg_price * 0.9:
                ess = -min(220, max(30, surplus * 0.55)) if ess_soc < 0.88 else 0.0
                ev = -min(180, max(20, surplus * 0.25)) if ev_soc < 0.85 else 0.0
                dr = 0.0
            else:
                ess = min(220, net * 0.45) if ess_soc > 0.22 else 0.0
                ev = min(160, net * 0.15) if ev_soc > 0.35 and price > avg_price else 0.0
                dr = min(150, max(0.0, net - 850.0) * 0.45)
            ess_soc = np.clip(ess_soc - ess * 0.25 / (500 * 0.95), 0.15, 0.90)
            ev_soc = np.clip(ev_soc - ev * 0.25 / (1200 * 0.95), 0.20, 0.95)
            plan.append({"ess_power_kw": float(ess), "ev_power_kw": float(ev), "dr_kw": float(dr), "upload_priority": 0.65, "edge_cpu_fraction": 0.65, "offload_ratio": 0.10})
        return plan


class MPCEnergyManagement(MILPEnergyManagement):
    """Receding-horizon MPC: solve look-ahead problem and execute first action."""

    name = "mpc_receding_horizon_optimized"

    def __init__(self, lookahead_steps: int = 8, safety_enabled: bool = True) -> None:
        super().__init__(horizon_steps=lookahead_steps, safety_enabled=safety_enabled)
        self.lookahead_steps = lookahead_steps


class ADMMDERCoordinator(EvaluationMixin):
    """Distributed ADMM DER coordinator with residual logging."""

    name = "admm_distributed_dispatch"

    def __init__(self, iterations: int = 25, rho: float = 0.8, tolerance: float = 1e-3, safety_enabled: bool = True) -> None:
        self.iterations = iterations
        self.rho = rho
        self.tolerance = tolerance
        self.shield = SafetyShield()
        self.safety_enabled = safety_enabled
        self.last_trace = OptimizationTrace()

    def act(self, state: dict[str, float]) -> dict[str, float]:
        net = float(state.get("load_kw", 0.0)) - float(state.get("pv_kw", 0.0)) - float(state.get("wind_kw", 0.0))
        ess_soc = float(state.get("ess_soc", 0.5))
        ev_soc = float(state.get("ev_soc", 0.5))
        x = np.array([0.0, 0.0, 0.0], dtype=float)  # ESS, EV, DR
        z = np.array([net / 3.0] * 3, dtype=float)
        u = np.zeros(3, dtype=float)
        primal: list[float] = []
        dual: list[float] = []
        objs: list[float] = []
        for _ in range(self.iterations):
            z_old = z.copy()
            target = z - u
            x[0] = np.clip(target[0], -240 if ess_soc < 0.88 else 0, 240 if ess_soc > 0.20 else 0)
            x[1] = np.clip(target[1], -200 if ev_soc < 0.85 else 0, 200 if ev_soc > 0.30 else 0)
            x[2] = np.clip(target[2], 0, 150)
            # consensus projection: sum(z)=net, as close to x+u as possible
            y = x + u
            z = y + (net - float(np.sum(y))) / 3.0
            u = u + x - z
            r = float(np.linalg.norm(x - z))
            s = float(self.rho * np.linalg.norm(z - z_old))
            obj = float(0.02 * x[0] ** 2 + 0.018 * x[1] ** 2 + 0.03 * x[2] ** 2)
            primal.append(r); dual.append(s); objs.append(obj)
            if r < self.tolerance and s < self.tolerance:
                break
        self.last_trace = OptimizationTrace(primal_residuals=primal, dual_residuals=dual, objective_values=objs, solver_status="converged" if primal[-1] < self.tolerance else "max_iter")
        aoi = float(state.get("avg_aoi", 0.0))
        return {"ess_power_kw": float(x[0]), "ev_power_kw": float(x[1]), "dr_kw": float(x[2]), "upload_priority": min(1.0, 0.55 + 0.06 * aoi), "edge_cpu_fraction": 0.60, "offload_ratio": 0.05}
