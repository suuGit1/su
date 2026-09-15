"""按事件时间推进的上行传输与 DT 计算队列，资源预算单位为 bit/s 和 cycle/s。"""
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import math
import numpy as np

CHANNELS=('ess','ev','dr_shift','dr_shed','pv','wind')
CONTRACT_VERSION='causal-c3-basic-dt-v1'


@dataclass
class CyberSpec:
    bandwidth_bps: float = 10000.
    cpu_cycles_per_second: float = 1.e7
    packet_overhead_bits: float = 1.e6
    update_cycles: float = 1.e9
    queue_capacity: int = 4
    packet_loss: float = 0.
    deadline_seconds: float = 900.
    joules_per_bit: float = 1.e-6
    joules_per_cycle: float = 1.e-9
    energy_price_per_kwh: float = .1

    @classmethod
    def load(cls,source=None):
        values=source if isinstance(source,dict) else json.loads(Path(source).read_text()) if source else {}
        obj=cls(**values)
        for k,v in asdict(obj).items():
            if not math.isfinite(v) or v<0:raise ValueError(k+' 必须为有限非负数')
        if type(obj.queue_capacity) is not int or obj.queue_capacity<1:raise ValueError('队列容量必须为正整数')
        if obj.packet_loss>1 or obj.deadline_seconds<=0 or obj.update_cycles<=0:raise ValueError('丢包概率、截止时间或计算量不合法')
        return obj


class CyberPipeline:
    def __init__(self,spec,seed):
        self.spec=spec;self.rng=np.random.default_rng(seed);self.now=0.;self.serial=0
        self.tx=[deque() for _ in CHANNELS];self.cpu=[deque() for _ in CHANNELS]

    def allocations(self,weights,budget):
        weights=np.asarray(weights,dtype=float)
        if weights.shape!=(6,) or not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('资源权重必须为六个有限非负数')
        # 闲置队列的配额不自动重分配，避免执行器暗中优化智能体动作。
        return weights/max(1.,float(weights.sum()))*budget

    def enqueue(self,payloads,upload):
        events=[]
        for k,enabled in enumerate(upload):
            if not enabled:continue
            if len(self.tx[k])>=self.spec.queue_capacity:
                events.append(dict(kind='tx_overflow',channel=k,time=self.now));continue
            payload=copy.deepcopy(payloads[k]);self.serial+=1
            bits=self.spec.packet_overhead_bits+8*len(json.dumps(payload,sort_keys=True).encode('utf-8'))
            p=dict(id=self.serial,channel=k,sampled=self.now,payload=payload,bits=bits,cycles=self.spec.update_cycles)
            self.tx[k].append(p)
            events.append(dict(kind='sample',channel=k,id=p['id'],time=self.now,sampled=self.now,bits=bits))
        return events

    def advance(self,seconds,bw_weights,cpu_weights):
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('推进时长必须为有限正数')
        bw=self.allocations(bw_weights,self.spec.bandwidth_bps)
        cpu=self.allocations(cpu_weights,self.spec.cpu_cycles_per_second)
        end=self.now+seconds;events=[];completed=[];bits_used=cycles_used=0.
        while self.now<end-1e-9:
            waits=[end-self.now]
            for k in range(6):
                if self.tx[k] and bw[k]>0:waits.append(self.tx[k][0]['bits']/bw[k])
                if self.cpu[k] and cpu[k]>0:waits.append(self.cpu[k][0]['cycles']/cpu[k])
            elapsed=float(max(0.,min(waits)))
            for k in range(6):
                if self.tx[k]:
                    n=min(self.tx[k][0]['bits'],bw[k]*elapsed);self.tx[k][0]['bits']-=n;bits_used+=n
                if self.cpu[k]:
                    n=min(self.cpu[k][0]['cycles'],cpu[k]*elapsed);self.cpu[k][0]['cycles']-=n;cycles_used+=n
            self.now+=elapsed
            for k in range(6):
                if self.cpu[k] and self.cpu[k][0]['cycles']<=1e-6:
                    p=self.cpu[k].popleft();p['finished']=self.now;completed.append(p)
                    events.append(dict(kind='dt_complete',channel=k,id=p['id'],sampled=p['sampled'],time=self.now,latency=self.now-p['sampled'],late=self.now-p['sampled']>self.spec.deadline_seconds))
                if self.tx[k] and self.tx[k][0]['bits']<=1e-6:
                    p=self.tx[k].popleft()
                    kind='tx_complete'
                    if self.rng.random()<self.spec.packet_loss:kind='packet_loss'
                    elif len(self.cpu[k])>=self.spec.queue_capacity:kind='cpu_overflow'
                    else:self.cpu[k].append(p)
                    events.append(dict(kind=kind,channel=k,id=p['id'],sampled=p['sampled'],time=self.now))
        self.now=end
        energy=bits_used*self.spec.joules_per_bit+cycles_used*self.spec.joules_per_cycle
        return completed,events,dict(tx_bits=bits_used,cpu_cycles=cycles_used,cyber_energy_j=energy,
            bandwidth_allocated_bps=bw.tolist(),cpu_allocated_cycles_per_second=cpu.tolist())

    def queue_state(self):
        return [sum(p['bits'] for p in q) for q in self.tx],[sum(p['cycles'] for p in q) for q in self.cpu]
