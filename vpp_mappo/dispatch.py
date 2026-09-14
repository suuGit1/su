"""统一资源参数、状态方程和经济费用；供调度器与学习环境共用。"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np


@dataclass
class DispatchSpec:
    capacities: tuple = (500.0, 1200.0)
    power_max: tuple = (250.0, 300.0)
    soc_min: tuple = (0.1, 0.2)
    soc_max: tuple = (0.9, 0.9)
    initial_soc: tuple = (0.55, 0.55)
    target_soc: tuple = (0.55, 0.55)
    efficiency: float = 0.95
    dr_max: float = 150.0
    grid_import: float = 5000.0
    grid_export: float = 3000.0
    sell_ratio: float = 0.65
    degradation: float = 0.01
    dr_cost: float = 0.03
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    assumed_line_ka: float = 0.4
    ess_bus: int = 17
    ev_bus: int = 24
    pv_buses: tuple = (12, 17, 21, 24, 28, 32)
    wind_bus: int = 32

    @classmethod
    def load(cls, path=None):
        obj = cls(**path) if isinstance(path, dict) else cls(**json.loads(Path(path).read_text(encoding='utf-8'))) if path else cls()
        for key in ('capacities', 'power_max', 'soc_min', 'soc_max', 'initial_soc', 'target_soc'):
            a = np.asarray(getattr(obj, key), dtype=float)
            if a.shape != (2,) or not np.isfinite(a).all():
                raise ValueError(key + ' 必须包含两个有限数')
        if not (np.asarray(obj.capacities) > 0).all() or not (np.asarray(obj.power_max) > 0).all():
            raise ValueError('容量与功率必须为正数')
        for i in range(2):
            if not 0 <= obj.soc_min[i] <= min(obj.initial_soc[i], obj.target_soc[i]) <= max(obj.initial_soc[i], obj.target_soc[i]) <= obj.soc_max[i] <= 1:
                raise ValueError('SOC 边界或目标不合法')
        for key in ('dr_max', 'grid_import', 'grid_export', 'degradation', 'dr_cost', 'assumed_line_ka'):
            if not np.isfinite(getattr(obj, key)) or getattr(obj, key) <= 0:
                raise ValueError(key + ' 必须为有限正数')
        if not 0 < obj.efficiency <= 1 or not 0 <= obj.sell_ratio <= 1 or not 0 < obj.voltage_min < 1 < obj.voltage_max:
            raise ValueError('效率、售电比例或电压范围不合法')
        if not obj.pv_buses or any(type(b) is not int or not 1 <= b <= 32 for b in [obj.ess_bus, obj.ev_bus, obj.wind_bus, *obj.pv_buses]):
            raise ValueError('资源母线使用 pandapower 的 1–32 索引，0 为平衡母线')
        return obj

    def next_soc(self, soc, action, dt):
        p = np.asarray(action[:2])
        return np.asarray(soc) + (np.maximum(-p, 0)*self.efficiency - np.maximum(p, 0)/self.efficiency)*dt/np.asarray(self.capacities)

    def cost(self, grid, action, price, dt):
        return float(dt*(price*(max(grid, 0)-self.sell_ratio*max(-grid, 0)) + self.degradation*np.abs(action[:2]).sum() + self.dr_cost*action[2]))

    def record(self):
        return asdict(self)
