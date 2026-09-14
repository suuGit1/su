"""Safety-shield layer for VPP energy-management actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyLimits:
    """Operational limits used by the DT-C3 safety shield."""

    dt_hours: float = 0.25
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    ess_capacity_kwh: float = 500.0
    ess_soc_min: float = 0.1
    ess_soc_max: float = 0.9
    ess_power_max_kw: float = 250.0
    ev_capacity_kwh: float = 1200.0
    ev_soc_min: float = 0.2
    ev_soc_target: float = 0.8
    ev_power_max_kw: float = 300.0
    grid_import_max_kw: float = 1200.0
    grid_export_max_kw: float = 800.0
    dr_max_kw: float = 150.0
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05


@dataclass
class ConstraintViolation:
    name: str
    amount: float
    message: str


@dataclass
class SafetyReport:
    original_action: dict[str, float]
    corrected_action: dict[str, float]
    violations: list[ConstraintViolation] = field(default_factory=list)
    safe: bool = True

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "violation_count": self.violation_count,
            "violations": [{"name": v.name, "amount": v.amount, "message": v.message} for v in self.violations],
            "corrected_action": self.corrected_action,
        }


class SafetyShield:
    """Rule/projection-based safety shield.

    Action convention:
    * ``ess_power_kw`` > 0: ESS discharges to support the VPP;
    * ``ess_power_kw`` < 0: ESS charges;
    * ``ev_power_kw`` follows the same sign convention;
    * ``dr_kw`` > 0: load reduction;
    * ``grid_power_kw`` > 0: grid import, < 0: grid export.
    """

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()

    def project(self, state: dict[str, float], action: dict[str, float]) -> SafetyReport:
        corrected = {k: float(v) for k, v in action.items()}
        original = dict(corrected)
        violations: list[ConstraintViolation] = []

        corrected["ess_power_kw"] = self._clip_storage_power(
            corrected.get("ess_power_kw", 0.0),
            float(state.get("ess_soc", 0.5)),
            self.limits.ess_capacity_kwh,
            self.limits.ess_soc_min,
            self.limits.ess_soc_max,
            self.limits.ess_power_max_kw,
            "ess",
            violations,
        )
        corrected["ev_power_kw"] = self._clip_storage_power(
            corrected.get("ev_power_kw", 0.0),
            float(state.get("ev_soc", 0.5)),
            self.limits.ev_capacity_kwh,
            self.limits.ev_soc_min,
            1.0,
            self.limits.ev_power_max_kw,
            "ev",
            violations,
        )
        corrected["dr_kw"] = self._clip(
            corrected.get("dr_kw", 0.0), 0.0, self.limits.dr_max_kw, "dr_kw", violations
        )

        net_load = (
            float(state.get("load_kw", 0.0))
            - float(state.get("pv_kw", 0.0))
            - float(state.get("wind_kw", 0.0))
            - corrected["ess_power_kw"]
            - corrected["ev_power_kw"]
            - corrected["dr_kw"]
        )
        grid_power = corrected.get("grid_power_kw", net_load)
        grid_power = self._clip(
            grid_power,
            -self.limits.grid_export_max_kw,
            self.limits.grid_import_max_kw,
            "grid_power_kw",
            violations,
        )
        corrected["grid_power_kw"] = grid_power

        return SafetyReport(
            original_action=original,
            corrected_action=corrected,
            violations=violations,
            safe=len(violations) == 0,
        )

    def check_state(self, state: dict[str, float]) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []
        ess_soc = float(state.get("ess_soc", 0.5))
        ev_soc = float(state.get("ev_soc", 0.5))
        if ess_soc < self.limits.ess_soc_min - 1e-9:
            violations.append(ConstraintViolation("ess_soc_min", self.limits.ess_soc_min - ess_soc, "ESS SOC below lower bound"))
        if ess_soc > self.limits.ess_soc_max + 1e-9:
            violations.append(ConstraintViolation("ess_soc_max", ess_soc - self.limits.ess_soc_max, "ESS SOC above upper bound"))
        if ev_soc < self.limits.ev_soc_min - 1e-9:
            violations.append(ConstraintViolation("ev_soc_min", self.limits.ev_soc_min - ev_soc, "EV SOC below lower bound"))
        if ev_soc > 1.0 + 1e-9:
            violations.append(ConstraintViolation("ev_soc_max", ev_soc - 1.0, "EV SOC 超过上限"))
        return violations

    def _clip_storage_power(
        self,
        power_kw: float,
        soc: float,
        capacity_kwh: float,
        soc_min: float,
        soc_max: float,
        pmax: float,
        prefix: str,
        violations: list[ConstraintViolation],
    ) -> float:
        power = self._clip(power_kw, -pmax, pmax, f"{prefix}_power_kw", violations)
        hours = self.limits.dt_hours
        if power > 0:  # discharge
            available_kw = max(0.0, (soc - soc_min) * capacity_kwh * self.limits.discharge_efficiency / hours)
            if power > available_kw:
                violations.append(ConstraintViolation(f"{prefix}_soc_min", power - available_kw, f"{prefix.upper()} discharge would violate SOC min"))
                power = available_kw
        elif power < 0:  # charge
            headroom_kw = max(0.0, (soc_max - soc) * capacity_kwh / (hours * self.limits.charge_efficiency))
            if abs(power) > headroom_kw:
                violations.append(ConstraintViolation(f"{prefix}_soc_max", abs(power) - headroom_kw, f"{prefix.upper()} charge would violate SOC max"))
                power = -headroom_kw
        return power

    def _clip(
        self,
        value: float,
        lo: float,
        hi: float,
        name: str,
        violations: list[ConstraintViolation],
    ) -> float:
        if value < lo:
            violations.append(ConstraintViolation(name, lo - value, f"{name} below lower bound"))
            return lo
        if value > hi:
            violations.append(ConstraintViolation(name, value - hi, f"{name} above upper bound"))
            return hi
        return value
