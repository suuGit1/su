"""会话级 EV、弃电及可削减/可转移 DR 的统一资源定义。"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import numpy as np
from .dispatch import DispatchSpec
from .network import Network33


@dataclass
class FlexSpec:
    ev_station_kw: float = 300.0
    dr_shift_kw: float = 100.0
    dr_repay_kw: float = 150.0
    dr_backlog_kwh: float = 300.0
    dr_shift_budget_kwh: float = 300.0
    dr_shed_kw: float = 50.0
    dr_shed_budget_kwh: float = 100.0
    dr_fraction: float = 0.2
    shift_cost: float = 0.01
    shed_cost: float = 0.1
    curtail_cost: float = 0.05

    @classmethod
    def load(cls, source=None):
        values=source if isinstance(source,dict) else json.loads(Path(source).read_text()) if source else {}
        obj=cls(**values)
        for k,v in asdict(obj).items():
            if not math.isfinite(v) or v < 0: raise ValueError(k+' 必须为有限非负数')
        if not 0 < obj.dr_fraction <= 1 or obj.ev_station_kw<=0: raise ValueError('DR 比例或充电站容量不合法')
        return obj

    def cost(self, base, grid, action, price, dt):
        e,ev,shift,shed,pv,wind=action
        return float(dt*(price*(max(grid,0)-base.sell_ratio*max(-grid,0))+base.degradation*abs(e)
                         +self.shift_cost*abs(shift)+self.shed_cost*shed+self.curtail_cost*(pv+wind)))


def validate_sessions(records,horizon,dt):
    result=[]; ids=set()
    for record in records:
        s=dict(record)
        if set(s)!={'id','arrival_step','departure_step','energy_kwh','max_kw'}: raise ValueError('EV 会话字段不完整或存在未知字段')
        if not isinstance(s['id'],str) or not s['id'] or s['id'] in ids: raise ValueError('EV 会话 id 必须唯一且非空')
        a,d=s['arrival_step'],s['departure_step']
        if type(a) is not int or type(d) is not int or not 0<=a<d<=horizon: raise ValueError('EV 时段必须满足 0≤接入<离站≤horizon')
        if not all(math.isfinite(s[k]) for k in ('energy_kwh','max_kw')) or s['energy_kwh']<0 or s['max_kw']<=0: raise ValueError('EV 电量或功率不合法')
        if s['energy_kwh']>s['max_kw']*(d-a)*dt+1e-7: raise ValueError('EV 需求超过自身可充电能力')
        ids.add(s['id']);result.append(s)
    return sorted(result,key=lambda s:(s['arrival_step'],s['departure_step'],s['id']))


def read_bundle(path):
    bundle=json.loads(Path(path).read_text(encoding='utf-8'))
    if bundle.get('schema_version')!=1 or bundle.get('energy_basis')!='grid_kwh' or not isinstance(bundle.get('scenarios'),dict) or not bundle.get('source'):
        raise ValueError('EV JSON 必须声明版本、来源、grid_kwh 电量口径及 scenarios')
    return bundle


def grid_power(row, action):
    return float(row['load_kw']-row['pv_kw']-row['wind_kw']+np.dot([-1,1,-1,-1,1,1],action))


class FlexNetwork(Network33):
    def __init__(self, spec):
        super().__init__(spec)
        self.pa=np.zeros((len(self.net.bus),6));self.qa=np.zeros((len(self.net.bus),6))
        self.pa[spec.ess_bus,0]=-0.001
        self.pa[spec.ev_bus,1]=0.001
        for k in (2,3):
            self.pa[:,k]=-self.weights/1000;self.qa[:,k]=-self.q_weights/1000
        for b in spec.pv_buses:self.pa[b,4]+=1/(1000*len(spec.pv_buses))
        self.pa[spec.wind_bus,5]=0.001
        self.pcoef=self.downstream@self.pa;self.qcoef=self.downstream@self.qa
        self.vcoef=-2*self.paths@(self.r[:,None]*self.pcoef+self.x[:,None]*self.qcoef)/self.kv**2
