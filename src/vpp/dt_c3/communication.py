"""Power-IoT communication models for DT-C3 VPP experiments.

The module intentionally stays dependency-light so that it can be used in
paper experiments without a full MQTT/OCPP simulator.  It emulates the main
non-ideal communication effects required by the DT-C3 research plan:

* discrete control-step delay;
* packet loss;
* device-offline events;
* information freshness / Age of Information (AoI).

The returned observations can be directly fed into the digital-twin state
synchronizer or the communication-aware RL policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
import random
import math


@dataclass
class CommunicationConfig:
    """Configuration for a non-ideal Power-IoT communication channel."""

    max_delay_steps: int = 0
    packet_loss_rate: float = 0.0
    offline_probability: float = 0.0
    seed: int | None = 42
    fixed_delay_steps: int | None = None
    bandwidth_kbps: float = 512.0
    packet_size_kb: float = 2.0
    upload_interval_steps: int = 1
    queue_delay_mean_steps: float = 0.0
    command_downlink_delay_steps: int = 0
    communication_cost_per_kb: float = 1e-5

    def __post_init__(self) -> None:
        if self.max_delay_steps < 0:
            raise ValueError("max_delay_steps must be non-negative")
        if not 0.0 <= self.packet_loss_rate <= 1.0:
            raise ValueError("packet_loss_rate must be within [0, 1]")
        if not 0.0 <= self.offline_probability <= 1.0:
            raise ValueError("offline_probability must be within [0, 1]")
        if self.fixed_delay_steps is not None and self.fixed_delay_steps < 0:
            raise ValueError("fixed_delay_steps must be non-negative")
        if self.bandwidth_kbps <= 0:
            raise ValueError("bandwidth_kbps must be positive")
        if self.packet_size_kb <= 0:
            raise ValueError("packet_size_kb must be positive")
        if self.upload_interval_steps <= 0:
            raise ValueError("upload_interval_steps must be positive")


@dataclass
class CommunicationState:
    """Communication state observed by the control algorithm."""

    device_id: str
    step: int
    delivered_step: int | None
    delay_steps: int
    aoi: int
    packet_lost: bool = False
    offline: bool = False
    missing: bool = False
    bandwidth_kbps: float = 512.0
    packet_size_kb: float = 2.0
    transmission_delay_steps: float = 0.0
    queue_delay_steps: float = 0.0
    command_delay_steps: int = 0
    communication_overhead_kb: float = 0.0
    communication_cost: float = 0.0
    delivered: bool = False

    @property
    def is_fresh(self) -> bool:
        return not self.missing and self.aoi == 0


@dataclass
class CommunicationPacket:
    """Packet emitted by a Power-IoT device."""

    device_id: str
    step: int
    payload: dict[str, Any]


@dataclass
class TransmissionResult:
    """Observed payload plus its communication metadata."""

    observation: dict[str, Any] | None
    communication: CommunicationState


class CommunicationChannel:
    """Discrete-time non-ideal communication channel.

    The channel stores all emitted packets in per-device buffers.  At step t,
    a controller receives a packet from t-delay if it exists and is not lost.
    If the packet is lost/offline, the last successfully delivered payload is
    reused when available and AoI increases.  If no payload is available, the
    observation is ``None`` and the digital twin must reconstruct the state.
    """

    def __init__(self, config: CommunicationConfig | None = None) -> None:
        self.config = config or CommunicationConfig()
        self._rng = random.Random(self.config.seed)
        self._runtime_bandwidth_kbps = self.config.bandwidth_kbps
        self._runtime_upload_interval_steps = self.config.upload_interval_steps
        self._runtime_packet_loss_multiplier = 1.0
        self._runtime_delay_multiplier = 1.0
        self._buffers: dict[str, list[CommunicationPacket]] = {}
        self._last_delivered: dict[str, CommunicationPacket] = {}
        self._history: list[CommunicationState] = []

    def reset(self) -> None:
        self._buffers.clear()
        self._last_delivered.clear()
        self._history.clear()
        self._rng = random.Random(self.config.seed)
        self._runtime_bandwidth_kbps = self.config.bandwidth_kbps
        self._runtime_upload_interval_steps = self.config.upload_interval_steps
        self._runtime_packet_loss_multiplier = 1.0
        self._runtime_delay_multiplier = 1.0

    def _sample_delay(self) -> int:
        if self.config.fixed_delay_steps is not None:
            return min(self.config.fixed_delay_steps, self.config.max_delay_steps)
        if self.config.max_delay_steps <= 0:
            return 0
        return self._rng.randint(0, self.config.max_delay_steps)

    def transmit(self, device_id: str, true_state: dict[str, Any], step: int) -> TransmissionResult:
        """Emit and receive one device packet at the current control step."""
        packet = CommunicationPacket(device_id=device_id, step=step, payload=copy.deepcopy(true_state))
        self._buffers.setdefault(device_id, []).append(packet)

        offline = self._rng.random() < self.config.offline_probability
        effective_loss = max(0.0, min(1.0, self.config.packet_loss_rate * self._runtime_packet_loss_multiplier))
        packet_lost = self._rng.random() < effective_loss
        # Uplink delay combines configured network delay, bandwidth-limited
        # serialization delay, random queueing delay and optional command delay.
        serialization_steps = self.config.packet_size_kb / max(1e-9, self._runtime_bandwidth_kbps)
        queue_delay = self._rng.expovariate(1.0 / self.config.queue_delay_mean_steps) if self.config.queue_delay_mean_steps > 0 else 0.0
        requested_delay = int(math.ceil((self._sample_delay() + int(math.ceil(queue_delay)) + self.config.command_downlink_delay_steps) * self._runtime_delay_multiplier))
        if step % self._runtime_upload_interval_steps != 0:
            requested_delay += step % self._runtime_upload_interval_steps
        target_step = step - requested_delay

        delivered: CommunicationPacket | None = None
        missing = False

        if offline or packet_lost or target_step < 0:
            delivered = self._last_delivered.get(device_id)
            missing = delivered is None
        else:
            candidates = [p for p in self._buffers[device_id] if p.step <= target_step]
            if candidates:
                delivered = max(candidates, key=lambda p: p.step)
                self._last_delivered[device_id] = delivered
            else:
                delivered = self._last_delivered.get(device_id)
                missing = delivered is None

        delivered_step = None if delivered is None else delivered.step
        aoi = step + 1 if delivered_step is None else max(0, step - delivered_step)
        state = CommunicationState(
            device_id=device_id,
            step=step,
            delivered_step=delivered_step,
            delay_steps=requested_delay,
            aoi=aoi,
            packet_lost=packet_lost,
            offline=offline,
            missing=missing,
            bandwidth_kbps=self._runtime_bandwidth_kbps,
            packet_size_kb=self.config.packet_size_kb,
            transmission_delay_steps=serialization_steps,
            queue_delay_steps=queue_delay,
            command_delay_steps=self.config.command_downlink_delay_steps,
            communication_overhead_kb=self.config.packet_size_kb,
            communication_cost=self.config.packet_size_kb * self.config.communication_cost_per_kb,
            delivered=delivered is not None and not missing,
        )
        self._history.append(state)
        return TransmissionResult(
            observation=None if delivered is None else copy.deepcopy(delivered.payload),
            communication=state,
        )

    def transmit_batch(self, true_states: dict[str, dict[str, Any]], step: int) -> dict[str, TransmissionResult]:
        """Transmit one packet for each device in a batch."""
        return {device_id: self.transmit(device_id, state, step) for device_id, state in true_states.items()}

    def average_aoi(self) -> float:
        if not self._history:
            return 0.0
        return sum(s.aoi for s in self._history) / len(self._history)

    def loss_ratio(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for s in self._history if s.packet_lost or s.offline or s.missing) / len(self._history)

    def delivery_ratio(self) -> float:
        if not self._history:
            return 1.0
        return sum(1 for s in self._history if s.delivered) / len(self._history)

    def average_delay(self) -> float:
        if not self._history:
            return 0.0
        return sum(s.delay_steps + s.transmission_delay_steps + s.queue_delay_steps + s.command_delay_steps for s in self._history) / len(self._history)

    def total_overhead_kb(self) -> float:
        return sum(s.communication_overhead_kb for s in self._history)

    def total_communication_cost(self) -> float:
        return sum(s.communication_cost for s in self._history)

    def stats(self) -> dict[str, float]:
        return {
            "avg_aoi": self.average_aoi(),
            "loss_ratio": self.loss_ratio(),
            "delivery_ratio": self.delivery_ratio(),
            "avg_delay_steps": self.average_delay(),
            "communication_overhead_kb": self.total_overhead_kb(),
            "communication_cost": self.total_communication_cost(),
        }


    def apply_c3_report(self, report: Any) -> None:
        """Apply C-C-C resource decisions to future transmissions.

        This is what makes communication decisions operational: upload priority
        and bandwidth allocation from the controller modify actual bandwidth,
        upload interval, loss probability and delay for later packets.
        """
        self._runtime_bandwidth_kbps = max(1.0, float(getattr(report, "adaptive_bandwidth_kbps", self.config.bandwidth_kbps)))
        self._runtime_upload_interval_steps = max(1, int(getattr(report, "adaptive_upload_interval_steps", self.config.upload_interval_steps)))
        self._runtime_packet_loss_multiplier = max(0.05, min(2.0, float(getattr(report, "packet_loss_multiplier", 1.0))))
        self._runtime_delay_multiplier = max(0.05, min(2.0, float(getattr(report, "delay_multiplier", 1.0))))

    def resource_state(self) -> dict[str, float]:
        return {
            "runtime_bandwidth_kbps": float(self._runtime_bandwidth_kbps),
            "runtime_upload_interval_steps": float(self._runtime_upload_interval_steps),
            "runtime_packet_loss_multiplier": float(self._runtime_packet_loss_multiplier),
            "runtime_delay_multiplier": float(self._runtime_delay_multiplier),
        }

    def history(self) -> list[CommunicationState]:
        return list(self._history)
