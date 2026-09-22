"""显式区分设备实测、仿真下行与假设计时；不把 Python 墙钟直接当作边缘 CPU 周期。"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TimingContract:
    downlink_bps:float=10000.
    command_bits:int=1024
    propagation_seconds:float=.02
    deadline_seconds:float=1.
    provenance:str='显式仿真假设，未校准至现场设备'

    def __post_init__(self):
        if not all(math.isfinite(v) and v>=0 for v in (self.downlink_bps,self.command_bits,self.propagation_seconds,self.deadline_seconds)) or self.deadline_seconds<=0:
            raise ValueError('计时参数非法')

    def latency(self,inference,safety,queue_wait=0.):
        if not all(math.isfinite(v) and v>=0 for v in (inference,safety,queue_wait)):raise ValueError('耗时必须有限非负')
        if self.downlink_bps==0:return dict(delivered=False,seconds=None,deadline_missed=True)
        total=inference+safety+queue_wait+self.command_bits/self.downlink_bps+self.propagation_seconds
        return dict(delivered=True,seconds=total,deadline_missed=total>self.deadline_seconds)


def summarize(samples):
    a=np.asarray(samples,dtype=float)
    if not len(a):return dict(n=0)
    if not np.isfinite(a).all() or (a<0).any():raise ValueError('耗时样本非法')
    return dict(n=len(a),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(a.max()))


class CommandQueue:
    """调度步边界上的下行命令队列；非零到达延迟会在下一边界才生效。"""
    def __init__(self,contract):self.contract=contract;self.pending=[]

    def submit(self,now,command,inference,safety):
        result=self.contract.latency(inference,safety)
        if result['delivered'] and not result['deadline_missed']:
            self.pending.append((now+result['seconds'],now,np.asarray(command,dtype=float).copy()))
        return result

    def receive(self,now):
        arrived=[x for x in self.pending if x[0]<=now and now-x[1]<=self.contract.deadline_seconds]
        expired=sum(now-x[1]>self.contract.deadline_seconds for x in self.pending)
        self.pending=[x for x in self.pending if x[0]>now and now-x[1]<=self.contract.deadline_seconds]
        if not arrived:return None,expired
        # 后发命令覆盖先发命令；策略看不到尚未到达的执行效果。
        return max(arrived,key=lambda x:x[1])[2],expired
