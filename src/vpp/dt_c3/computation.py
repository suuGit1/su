"""Edge-computation scheduling model for DT-C3 experiments.

The purpose of this module is to turn the "computation" part of C-C-C from
an abstract penalty into an explicit resource-allocation model.  It tracks
three online tasks required by the proposed paper pipeline:

* digital-twin state update;
* neural policy inference;
* safety-shield / twin-preview verification.

The scheduler supports local edge execution, cloud offloading, queueing delay,
CPU-frequency limits and computation-budget violations.  It is intentionally
compact and dependency-light so that it can run in the user's Windows/Python
research environment without a full MEC simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


@dataclass
class EdgeTask:
    name: str
    cycles_m: float
    data_kb: float = 1.0
    deadline_s: float = 0.05
    priority: float = 0.5


@dataclass
class EdgeNodeConfig:
    edge_cpu_ghz: float = 2.4
    cloud_cpu_ghz: float = 12.0
    edge_queue_capacity_mcycles: float = 180.0
    cloud_roundtrip_s: float = 0.035
    offload_bandwidth_kbps: float = 2048.0
    energy_per_mcycle: float = 2e-6
    offload_cost_per_kb: float = 2e-5
    latency_budget_s: float = 0.05


@dataclass
class ComputationReport:
    edge_cpu_fraction: float
    offload_ratio: float
    total_cycles_m: float
    edge_cycles_m: float
    cloud_cycles_m: float
    queue_delay_s: float
    execution_latency_s: float
    communication_latency_s: float
    total_latency_s: float
    computation_cost: float
    offloading_cost: float
    deadline_violations: int
    effective_dt_update_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "edge_cpu_fraction": self.edge_cpu_fraction,
            "offload_ratio": self.offload_ratio,
            "total_cycles_m": self.total_cycles_m,
            "edge_cycles_m": self.edge_cycles_m,
            "cloud_cycles_m": self.cloud_cycles_m,
            "queue_delay_s": self.queue_delay_s,
            "execution_latency_s": self.execution_latency_s,
            "communication_latency_s": self.communication_latency_s,
            "total_latency_s": self.total_latency_s,
            "computation_cost": self.computation_cost,
            "offloading_cost": self.offloading_cost,
            "deadline_violations": self.deadline_violations,
            "effective_dt_update_rate": self.effective_dt_update_rate,
        }


class EdgeComputationScheduler:
    """Queue-aware edge/cloud scheduler used by the C3 resource manager."""

    def __init__(self, config: EdgeNodeConfig | None = None) -> None:
        self.config = config or EdgeNodeConfig()
        self.queue_backlog_mcycles = 0.0

    def reset(self) -> None:
        self.queue_backlog_mcycles = 0.0

    def evaluate(
        self,
        tasks: list[EdgeTask],
        edge_cpu_fraction: float,
        offload_ratio: float = 0.0,
        allocated_bandwidth_kbps: float | None = None,
    ) -> ComputationReport:
        cfg = self.config
        edge_cpu_fraction = max(0.05, min(1.0, float(edge_cpu_fraction)))
        offload_ratio = max(0.0, min(0.95, float(offload_ratio)))
        bandwidth = max(1.0, float(allocated_bandwidth_kbps or cfg.offload_bandwidth_kbps))

        total_cycles = sum(max(0.0, t.cycles_m) for t in tasks)
        total_data = sum(max(0.0, t.data_kb) for t in tasks)
        cloud_cycles = total_cycles * offload_ratio
        edge_cycles = total_cycles - cloud_cycles

        # Backlog is processed by the allocated CPU.  A larger edge allocation
        # shortens both queueing and execution latency.
        effective_edge_ghz = cfg.edge_cpu_ghz * edge_cpu_fraction
        edge_service_mcycles_per_s = max(1e-6, effective_edge_ghz * 1000.0)
        queue_delay = self.queue_backlog_mcycles / edge_service_mcycles_per_s
        edge_latency = edge_cycles / edge_service_mcycles_per_s
        cloud_latency = 0.0
        if cloud_cycles > 0:
            cloud_latency = cfg.cloud_roundtrip_s + cloud_cycles / (cfg.cloud_cpu_ghz * 1000.0)
        comm_latency = (total_data * offload_ratio) / bandwidth
        total_latency = queue_delay + max(edge_latency, cloud_latency + comm_latency)

        # Update queue backlog for the next control period.  A 15-minute period
        # is long, but online control/DT tasks are fast; this queue mainly tests
        # computational scarcity scenarios.
        processed = edge_service_mcycles_per_s * cfg.latency_budget_s
        self.queue_backlog_mcycles = max(0.0, self.queue_backlog_mcycles + edge_cycles - processed)
        self.queue_backlog_mcycles = min(self.queue_backlog_mcycles, cfg.edge_queue_capacity_mcycles)

        deadline_violations = sum(1 for t in tasks if total_latency > t.deadline_s)
        comp_cost = edge_cycles * cfg.energy_per_mcycle + cloud_cycles * cfg.energy_per_mcycle * 0.35
        offload_cost = total_data * offload_ratio * cfg.offload_cost_per_kb
        dt_update_rate = 1.0 / max(1.0, 1.0 + self.queue_backlog_mcycles / max(1e-6, total_cycles + 1e-6))

        return ComputationReport(
            edge_cpu_fraction=edge_cpu_fraction,
            offload_ratio=offload_ratio,
            total_cycles_m=total_cycles,
            edge_cycles_m=edge_cycles,
            cloud_cycles_m=cloud_cycles,
            queue_delay_s=queue_delay,
            execution_latency_s=max(edge_latency, cloud_latency),
            communication_latency_s=comm_latency,
            total_latency_s=total_latency,
            computation_cost=comp_cost,
            offloading_cost=offload_cost,
            deadline_violations=deadline_violations,
            effective_dt_update_rate=dt_update_rate,
        )


def default_dt_c3_tasks(dt_risk_count: int = 0, horizon_steps: int = 4) -> list[EdgeTask]:
    """Default online computation workload used in every control step."""
    return [
        EdgeTask("digital_twin_update", cycles_m=28.0 + 2.0 * horizon_steps, data_kb=7.5, deadline_s=0.045, priority=0.9),
        EdgeTask("policy_inference", cycles_m=9.0, data_kb=1.5, deadline_s=0.030, priority=0.8),
        EdgeTask("safety_preview", cycles_m=14.0 + 3.0 * dt_risk_count, data_kb=2.0, deadline_s=0.050, priority=1.0),
    ]
