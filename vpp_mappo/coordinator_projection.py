"""由固定公开观察推导服务下限，actor决定带宽与CPU各自的剩余额度。"""
import numpy as np
from .coordinator_v2 import MaskedRiskCoordinator


class PublicServiceProjection(MaskedRiskCoordinator):
    def __init__(self,config,dispatch,flex,cyber):
        super().__init__(config,dispatch,flex,config.coordinator_active_channels,mapped=True)
        self.cyber=cyber

    def allocate(self,observation,bw,cpu):
        # 映射风险回退仍保留actor输入；不读取环境队列对象或物理传感器。
        fallback_bw,fallback_cpu,record=super().allocate(observation,bw,cpu)
        if self.config.coordinator_mode!='schedule':return fallback_bw,fallback_cpu,record
        x=np.asarray(observation);s=self.cyber;T=self.config.dt_hours*3600
        B=s.bandwidth_bps
        control=min(self.config.control_cycles,s.cpu_cycles_per_second*T) if self.config.command_timing else 0.
        C=max(0.,s.cpu_cycles_per_second-control/T)
        tx=np.maximum(0,x[24:30])*max(1.,s.packet_overhead_bits)
        cp=np.maximum(0,x[30:36])*s.update_cycles
        # 一个部分传输报文加后续完整报文的保守数量上界；不新增队列计数观察。
        count=np.where(tx>0,np.minimum(s.queue_capacity,np.ceil(tx/s.packet_overhead_bits)+1),0)
        A=(tx+s.packet_overhead_bits+self.config.coordinator_payload_bound_bits)*self.active
        D=(cp+(count+1)*s.update_cycles)*self.active
        duration=float(A.sum()/B+D.sum()/C) if min(B,C)>0 else float('inf')
        feasible=duration<=T
        floor_b=np.zeros(6);floor_c=np.zeros(6)
        out_b,out_c=fallback_bw,fallback_cpu
        if feasible:
            alpha=(A.sum()/B)/duration
            floor_b=A/(alpha*T*B);floor_c=D/((1-alpha)*T*C)
            def project(values,floor):
                values=np.asarray(values,dtype=float)
                if values.shape!=(6,) or not np.isfinite(values).all() or np.any(values<0):raise ValueError('actor资源权重须有限非负')
                values=np.where(self.active,values,0.)
                proportions=values/values.sum() if values.sum()>0 else self.active/self.active.sum()
                # 默认0保持旧行为；只分配服务下限之外的余量，不撤销完成下界。
                mix=self.config.coordinator_priority_slack_fraction
                priority=np.asarray(record['priority'],dtype=float);priority=priority/priority.sum()
                proportions=(1-mix)*proportions+mix*priority
                return floor+max(0.,1-floor.sum())*proportions
            out_b=project(bw,floor_b);out_c=project(cpu,floor_c)
        record.update(priority_slack_fraction=self.config.coordinator_priority_slack_fraction,contract='public-80-service-projection-v1',projection_applied=bool(feasible),
            predicted_serial_seconds=duration if np.isfinite(duration) else None,
            service_deadline_seconds=T,bandwidth_floor=floor_b.tolist(),cpu_floor=floor_c.tolist(),
            actor_bandwidth_slack=float(max(0,1-floor_b.sum())) if feasible else None,
            actor_cpu_slack=float(max(0,1-floor_c.sum())) if feasible else None,
            requested_bandwidth=np.asarray(bw).tolist(),requested_cpu=np.asarray(cpu).tolist(),
            input_contract='原80维公开观察的队列总量及已声明静态容量；无额外队列计数',
            service_scope='载荷在上界内且没有丢包或溢出的周期内充分条件；不可行时回退映射风险分配')
        return out_b,out_c,record
