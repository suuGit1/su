"""Reproducible DT-C3 experiment suite.

The suite maps directly to the six mandatory paper experiments:
1. normal communication;
2. communication delay sensitivity;
3. packet-loss sensitivity;
4. prediction-error robustness;
5. extreme disturbances;
6. ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import json

from .algorithms import (
    DTC3SafeMARL,
    NoCommunicationAwareEMS,
    NoSafetyShieldEMS,
    RandomEMS,
    RuleBasedEMS,
)
from .communication import CommunicationConfig
from .environment import PowerIoTVPPEnvironment, VPPEnvConfig
from .safety import SafetyLimits


@dataclass
class ExperimentSuiteConfig:
    horizon_steps: int = 96
    seed: int = 42
    output_dir: str = "output/dt_c3"
    grid_enabled: bool = True


@dataclass
class ExperimentSuiteResult:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, scenario: str, params: dict[str, Any], metrics: dict[str, Any]) -> None:
        row = {"scenario": scenario, **params, **metrics}
        self.rows.append(row)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("", encoding="utf-8")
            return
        keys = sorted({k for row in self.rows for k in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.rows, indent=2, ensure_ascii=False), encoding="utf-8")


class DTC3ExperimentSuite:
    def __init__(self, config: ExperimentSuiteConfig | None = None) -> None:
        self.config = config or ExperimentSuiteConfig()
        self.result = ExperimentSuiteResult()

    def make_env(
        self,
        delay_steps: int = 0,
        packet_loss: float = 0.0,
        prediction_error: float = 0.0,
        extreme_event: str | None = None,
    ) -> PowerIoTVPPEnvironment:
        communication = CommunicationConfig(
            max_delay_steps=delay_steps,
            fixed_delay_steps=delay_steps,
            packet_loss_rate=packet_loss,
            seed=self.config.seed,
        )
        return PowerIoTVPPEnvironment(
            VPPEnvConfig(
                horizon_steps=self.config.horizon_steps,
                seed=self.config.seed,
                communication=communication,
                prediction_error=prediction_error,
                use_grid_constraints=self.config.grid_enabled,
                safety_limits=SafetyLimits(),
                extreme_event=extreme_event,
            )
        )

    def run_normal_communication(self) -> None:
        methods = [DTC3SafeMARL(), RuleBasedEMS(), RandomEMS()]
        for method in methods:
            metrics = method.evaluate(self.make_env())
            self.result.add("normal_communication", {"method": method.name}, metrics)

    def run_delay_sensitivity(self, delays: list[int] | None = None) -> None:
        for delay in delays or [0, 1, 3, 5]:
            method = DTC3SafeMARL()
            metrics = method.evaluate(self.make_env(delay_steps=delay))
            self.result.add("delay_sensitivity", {"method": method.name, "delay_steps": delay}, metrics)

    def run_packet_loss_sensitivity(self, rates: list[float] | None = None) -> None:
        for rate in rates or [0.0, 0.05, 0.10, 0.20, 0.30]:
            method = DTC3SafeMARL()
            metrics = method.evaluate(self.make_env(packet_loss=rate))
            self.result.add("packet_loss_sensitivity", {"method": method.name, "packet_loss_rate": rate}, metrics)

    def run_prediction_error(self, errors: list[float] | None = None) -> None:
        for error in errors or [0.05, 0.10, 0.20]:
            method = DTC3SafeMARL()
            metrics = method.evaluate(self.make_env(prediction_error=error, delay_steps=1, packet_loss=0.05))
            self.result.add("prediction_error", {"method": method.name, "prediction_error": error}, metrics)

    def run_extreme_disturbance(self, events: list[str] | None = None) -> None:
        for event in events or ["pv_drop", "load_spike", "price_spike"]:
            method = DTC3SafeMARL()
            metrics = method.evaluate(self.make_env(extreme_event=event, delay_steps=1, packet_loss=0.05))
            self.result.add("extreme_disturbance", {"method": method.name, "event": event}, metrics)

    def run_ablation(self) -> None:
        methods = [DTC3SafeMARL(), NoCommunicationAwareEMS(), NoSafetyShieldEMS(), RuleBasedEMS()]
        for method in methods:
            metrics = method.evaluate(self.make_env(delay_steps=3, packet_loss=0.20, prediction_error=0.10))
            self.result.add("ablation", {"method": method.name}, metrics)

    def run_all(self) -> ExperimentSuiteResult:
        self.result = ExperimentSuiteResult()
        self.run_normal_communication()
        self.run_delay_sensitivity()
        self.run_packet_loss_sensitivity()
        self.run_prediction_error()
        self.run_extreme_disturbance()
        self.run_ablation()
        return self.result

    def save(self) -> tuple[Path, Path]:
        out = Path(self.config.output_dir)
        csv_path = out / "dt_c3_results.csv"
        json_path = out / "dt_c3_results.json"
        self.result.to_csv(csv_path)
        self.result.to_json(json_path)
        return csv_path, json_path
