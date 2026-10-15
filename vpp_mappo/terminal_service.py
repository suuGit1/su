"""公共终端库存服务：可达性仅考虑储能功率，不冒充网络可行证明。"""
import numpy as np


def band(config,spec):
    if config.terminal_soc_mode=='legacy_penalty':return None
    target=spec.target_soc[0];tol=config.terminal_soc_tolerance
    return max(spec.soc_min[0],target-tol),min(spec.soc_max[0],target+tol)


def reachable_band(spec,terminal,remaining_steps,dt):
    low,high=terminal;hours=max(0,remaining_steps)*dt
    return (max(spec.soc_min[0],low-spec.power_max[0]*spec.efficiency*hours/spec.capacities[0]),
            min(spec.soc_max[0],high+spec.power_max[0]*hours/(spec.efficiency*spec.capacities[0])))


def power_bounds(spec,soc_low,soc_high,next_band,dt):
    # 对整个SOC区间使用同一个动作；功率正号代表放电。
    low,high=next_band;cap=spec.capacities[0];eta=spec.efficiency
    def inverse(delta):return -delta*cap/(eta*dt) if delta>=0 else -delta*cap*eta/dt
    return max(-spec.power_max[0],inverse(high-soc_high)),min(spec.power_max[0],inverse(low-soc_low))


def shortfall_kwh(soc,terminal,capacity):
    if terminal is None:return 0.
    return float(max(terminal[0]-soc,soc-terminal[1],0)*capacity)
