"""IEEE 33-bus grid interface for DT-C3 VPP experiments.

The module exposes a pandapower-friendly interface but also includes a
lightweight deterministic fallback so tests and paper experiments can run
without optional power-flow packages installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

try:  # pragma: no cover - optional dependency
    import pandapower as pp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pp = None


@dataclass
class GridConfig:
    bus_count: int = 33
    base_mva: float = 10.0
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    line_loading_max_pct: float = 100.0
    slack_voltage_pu: float = 1.0


@dataclass
class PowerFlowResult:
    success: bool
    bus_voltage_pu: dict[int, float]
    line_loading_pct: dict[int, float]
    violations: list[str] = field(default_factory=list)
    backend: str = "simplified"

    @property
    def min_voltage_pu(self) -> float:
        return min(self.bus_voltage_pu.values()) if self.bus_voltage_pu else 1.0

    @property
    def max_line_loading_pct(self) -> float:
        return max(self.line_loading_pct.values()) if self.line_loading_pct else 0.0


class IEEE33BusInterface:
    """Radial IEEE 33-bus interface with optional pandapower backend."""

    def __init__(self, config: GridConfig | None = None, prefer_pandapower: bool = False) -> None:
        self.config = config or GridConfig()
        self.prefer_pandapower = prefer_pandapower and pp is not None
        self.backend = "pandapower" if self.prefer_pandapower else "simplified"
        self._line_edges = self._default_ieee33_edges()

    @property
    def has_pandapower(self) -> bool:
        return pp is not None

    def run_power_flow(self, injections_kw: dict[int, float]) -> PowerFlowResult:
        """Run a grid-constraint check.

        ``injections_kw`` uses positive values for generation/export at a bus
        and negative values for load/import at a bus.
        """
        if self.prefer_pandapower and pp is not None:  # pragma: no cover - optional dependency
            try:
                return self._run_pandapower(injections_kw)
            except Exception as exc:
                fallback = self._run_simplified(injections_kw)
                fallback.violations.append(f"pandapower_fallback:{exc}")
                return fallback
        return self._run_simplified(injections_kw)

    def check_constraints(self, injections_kw: dict[int, float]) -> list[str]:
        return self.run_power_flow(injections_kw).violations

    def vpp_to_bus_injections(
        self,
        state: dict[str, float],
        action: dict[str, float],
        bus_map: dict[str, int] | None = None,
    ) -> dict[int, float]:
        """Map aggregate VPP state/action to bus-level injections."""
        bus_map = bus_map or {"pv": 18, "wind": 22, "ess": 25, "ev": 30, "load": 7}
        injections = {i: 0.0 for i in range(1, self.config.bus_count + 1)}
        injections[bus_map["pv"]] += float(state.get("pv_kw", 0.0))
        injections[bus_map["wind"]] += float(state.get("wind_kw", 0.0))
        injections[bus_map["ess"]] += float(action.get("ess_power_kw", 0.0))
        injections[bus_map["ev"]] += float(action.get("ev_power_kw", 0.0))
        # Load is negative injection. DR reduces the local load.
        injections[bus_map["load"]] -= max(0.0, float(state.get("load_kw", 0.0)) - float(action.get("dr_kw", 0.0)))
        return injections

    def _run_simplified(self, injections_kw: dict[int, float]) -> PowerFlowResult:
        # Simplified radial feeder voltage-drop model.  It is not a substitute
        # for pandapower/OpenDSS in the final paper, but it provides a stable
        # interface and testable grid-safety logic.
        n = self.config.bus_count
        net_load = {bus: -float(injections_kw.get(bus, 0.0)) for bus in range(1, n + 1)}
        voltages = {1: self.config.slack_voltage_pu}
        line_loading: dict[int, float] = {}
        downstream = self._downstream_loads(net_load)

        for idx, (src, dst, r_pu) in enumerate(self._line_edges, start=1):
            flow_kw = downstream.get(dst, 0.0)
            drop = r_pu * flow_kw / 1000.0
            parent_v = voltages.get(src, self.config.slack_voltage_pu)
            # Generation can cause a small voltage rise; limit numerical drift.
            voltages[dst] = max(0.85, min(1.10, parent_v - drop))
            line_loading[idx] = min(200.0, abs(flow_kw) / 10.0)

        violations: list[str] = []
        for bus, v in voltages.items():
            if v < self.config.voltage_min_pu:
                violations.append(f"bus_{bus}_undervoltage:{v:.4f}")
            elif v > self.config.voltage_max_pu:
                violations.append(f"bus_{bus}_overvoltage:{v:.4f}")
        for line, loading in line_loading.items():
            if loading > self.config.line_loading_max_pct:
                violations.append(f"line_{line}_overload:{loading:.2f}")

        return PowerFlowResult(
            success=True,
            bus_voltage_pu=voltages,
            line_loading_pct=line_loading,
            violations=violations,
            backend="simplified",
        )

    def _downstream_loads(self, net_load: dict[int, float]) -> dict[int, float]:
        children: dict[int, list[int]] = {}
        for src, dst, _ in self._line_edges:
            children.setdefault(src, []).append(dst)

        def subtree(bus: int) -> float:
            return net_load.get(bus, 0.0) + sum(subtree(child) for child in children.get(bus, []))

        return {bus: subtree(bus) for bus in range(1, self.config.bus_count + 1)}

    def _default_ieee33_edges(self) -> list[tuple[int, int, float]]:
        # The topology follows the common IEEE 33-bus radial feeder ordering.
        # r_pu values are normalized sensitivities for the simplified backend.
        base = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
            (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
            (16, 17), (17, 18), (2, 19), (19, 20), (20, 21), (21, 22), (3, 23),
            (23, 24), (24, 25), (6, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (30, 31), (31, 32), (32, 33),
        ]
        return [(src, dst, 0.0015 + 0.0002 * math.sin(i)) for i, (src, dst) in enumerate(base, start=1)]

    def _run_pandapower(self, injections_kw: dict[int, float]) -> PowerFlowResult:  # pragma: no cover - optional dependency
        net = pp.create_empty_network(sn_mva=self.config.base_mva)
        buses = [pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}") for i in range(1, self.config.bus_count + 1)]
        pp.create_ext_grid(net, buses[0], vm_pu=self.config.slack_voltage_pu)
        for src, dst, _ in self._line_edges:
            pp.create_line_from_parameters(
                net,
                buses[src - 1],
                buses[dst - 1],
                length_km=1.0,
                r_ohm_per_km=0.3,
                x_ohm_per_km=0.2,
                c_nf_per_km=0.0,
                max_i_ka=0.4,
            )
        for bus, injection_kw in injections_kw.items():
            if bus < 1 or bus > self.config.bus_count:
                continue
            if injection_kw >= 0:
                pp.create_sgen(net, buses[bus - 1], p_mw=injection_kw / 1000.0, q_mvar=0.0)
            else:
                pp.create_load(net, buses[bus - 1], p_mw=abs(injection_kw) / 1000.0, q_mvar=0.0)
        pp.runpp(net)
        voltages = {i + 1: float(v) for i, v in enumerate(net.res_bus.vm_pu.values)}
        line_loading = {i + 1: float(v) for i, v in enumerate(net.res_line.loading_percent.values)}
        violations = []
        for bus, v in voltages.items():
            if v < self.config.voltage_min_pu:
                violations.append(f"bus_{bus}_undervoltage:{v:.4f}")
            elif v > self.config.voltage_max_pu:
                violations.append(f"bus_{bus}_overvoltage:{v:.4f}")
        for line, loading in line_loading.items():
            if loading > self.config.line_loading_max_pct:
                violations.append(f"line_{line}_overload:{loading:.2f}")
        return PowerFlowResult(True, voltages, line_loading, violations, "pandapower")
