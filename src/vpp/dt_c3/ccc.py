"""Communication-computation-control co-design utilities.

This upgraded version makes C-C-C decisions operational instead of merely
adding penalties.  Upload priority controls bandwidth, upload interval and
loss/delay reduction; edge CPU and offloading control computation latency; and
the digital-twin preview risk is coupled into the safety/control penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .computation import EdgeComputationScheduler, EdgeNodeConfig, default_dt_c3_tasks


@dataclass
class C3Config:
    base_bandwidth_kbps: float = 512.0
    max_extra_bandwidth_kbps: float = 1536.0
    min_upload_interval_steps: int = 1
    max_upload_interval_steps: int = 4
    min_packet_loss_multiplier: float = 0.35
    min_delay_multiplier: float = 0.40
    control_latency_budget_steps: float = 1.5
    bandwidth_cost_per_kbps: float = 1e-5
    control_risk_weight: float = 3.0
    edge: EdgeNodeConfig = field(default_factory=EdgeNodeConfig)


@dataclass
class C3Decision:
    upload_priority: float = 0.5
    edge_cpu_fraction: float = 0.5
    offload_ratio: float = 0.0
    control_aggressiveness: float = 0.5

    @classmethod
    def from_action(cls, action: dict[str, float]) -> "C3Decision":
        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))
        return cls(
            upload_priority=clamp01(action.get("upload_priority", 0.5)),
            edge_cpu_fraction=max(0.05, clamp01(action.get("edge_cpu_fraction", 0.5))),
            offload_ratio=clamp01(action.get("offload_ratio", 0.0)),
            control_aggressiveness=clamp01(action.get("control_aggressiveness", 0.5)),
        )


@dataclass
class C3Report:
    adaptive_bandwidth_kbps: float
    adaptive_upload_interval_steps: int
    packet_loss_multiplier: float
    delay_multiplier: float
    computation_latency_s: float
    computation_queue_delay_s: float
    offload_ratio: float
    computation_cost: float
    offloading_cost: float
    communication_resource_cost: float
    control_risk_penalty: float
    c3_total_penalty: float
    latency_budget_violation: int
    deadline_violations: int
    effective_dt_update_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "adaptive_bandwidth_kbps": self.adaptive_bandwidth_kbps,
            "adaptive_upload_interval_steps": self.adaptive_upload_interval_steps,
            "packet_loss_multiplier": self.packet_loss_multiplier,
            "delay_multiplier": self.delay_multiplier,
            "computation_latency_s": self.computation_latency_s,
            "computation_queue_delay_s": self.computation_queue_delay_s,
            "offload_ratio": self.offload_ratio,
            "computation_cost": self.computation_cost,
            "offloading_cost": self.offloading_cost,
            "communication_resource_cost": self.communication_resource_cost,
            "control_risk_penalty": self.control_risk_penalty,
            "c3_total_penalty": self.c3_total_penalty,
            "latency_budget_violation": self.latency_budget_violation,
            "deadline_violations": self.deadline_violations,
            "effective_dt_update_rate": self.effective_dt_update_rate,
        }


class C3ResourceManager:
    """Maps controller C3 decisions to physical comm/compute effects."""

    def __init__(self, config: C3Config | None = None) -> None:
        self.config = config or C3Config()
        self.scheduler = EdgeComputationScheduler(self.config.edge)

    def reset(self) -> None:
        self.scheduler.reset()

    def evaluate(
        self,
        action: dict[str, float],
        observed_state: dict[str, float],
        twin_preview: dict[str, Any] | None = None,
    ) -> C3Report:
        decision = C3Decision.from_action(action)
        cfg = self.config
        bandwidth = cfg.base_bandwidth_kbps + decision.upload_priority * cfg.max_extra_bandwidth_kbps
        interval_span = cfg.max_upload_interval_steps - cfg.min_upload_interval_steps
        upload_interval = int(round(cfg.max_upload_interval_steps - decision.upload_priority * interval_span))
        upload_interval = max(cfg.min_upload_interval_steps, min(cfg.max_upload_interval_steps, upload_interval))
        packet_loss_multiplier = 1.0 - decision.upload_priority * (1.0 - cfg.min_packet_loss_multiplier)
        delay_multiplier = 1.0 - decision.upload_priority * (1.0 - cfg.min_delay_multiplier)

        preview_risk = 0.0 if twin_preview is None or twin_preview.get("safe", True) else float(len(twin_preview.get("risks", [])))
        tasks = default_dt_c3_tasks(int(preview_risk), int((twin_preview or {}).get("horizon_steps", 4)))
        comp = self.scheduler.evaluate(tasks, decision.edge_cpu_fraction, decision.offload_ratio, bandwidth)

        avg_aoi = float(observed_state.get("avg_aoi", 0.0))
        loss = float(observed_state.get("packet_loss_ratio", 0.0))
        aggressiveness = abs(float(action.get("ess_power_kw", 0.0))) + abs(float(action.get("ev_power_kw", 0.0))) + abs(float(action.get("dr_kw", 0.0)))
        control_risk = cfg.control_risk_weight * (0.05 * avg_aoi + loss + 0.25 * preview_risk) * (1.0 + aggressiveness / 1000.0)
        communication_cost = bandwidth * cfg.bandwidth_cost_per_kbps
        latency_violation = int(comp.total_latency_s * 4.0 > cfg.control_latency_budget_steps)
        total = comp.computation_cost + comp.offloading_cost + communication_cost + control_risk + 2.0 * latency_violation + 0.5 * comp.deadline_violations
        return C3Report(
            adaptive_bandwidth_kbps=bandwidth,
            adaptive_upload_interval_steps=upload_interval,
            packet_loss_multiplier=packet_loss_multiplier,
            delay_multiplier=delay_multiplier,
            computation_latency_s=comp.total_latency_s,
            computation_queue_delay_s=comp.queue_delay_s,
            offload_ratio=decision.offload_ratio,
            computation_cost=comp.computation_cost,
            offloading_cost=comp.offloading_cost,
            communication_resource_cost=communication_cost,
            control_risk_penalty=control_risk,
            c3_total_penalty=total,
            latency_budget_violation=latency_violation,
            deadline_violations=comp.deadline_violations,
            effective_dt_update_rate=comp.effective_dt_update_rate,
        )
