"""公开队列驱动的两阶段服务分配；完成上界不包含随机丢包保证。"""
import numpy as np
from .coordinator_v2 import MaskedRiskCoordinator


class ServiceCoordinator(MaskedRiskCoordinator):
    def __init__(self, config, dispatch, flex, active_channels, value_priority=False):
        super().__init__(config, dispatch, flex, active_channels, mapped=True)
        self.value_priority=value_priority
        self.service_state=None
        self.public_state=None

    def assess(self, observation):
        record=super().assess(observation)
        if self.value_priority:
            x=np.asarray(self.public_state,dtype=float)
            if x.shape!=(80,) or not np.isfinite(x).all():raise ValueError('价值代理要求完整公开80维观察')
            w=np.asarray(self.interval_width);dt=self.config.dt_hours
            # 一步功率平衡的局部误差传播代理；不等于实际控制收益或最优信息价值。
            energy=np.array([2*w[0]*self.dispatch.capacities[0],2*w[1]*1000,
                             2*w[5]*5000*dt,2*w[4]*self.flex.dr_shed_budget_kwh,
                             2*w[7]*1000*dt,2*w[8]*1000*dt])
            cost=energy*abs(x[10]*.3)
            cost[2]+=2*w[6]*.3*abs(x[13]*5000)*dt
            carbon=energy*max(0,x[64])
            reserve=energy.copy()
            values=np.array([cost,carbon,reserve])/np.asarray(self.config.objective_scales)[:,None]
            task=self.config.task_focus
            weights=np.ones(3)/3  # 平衡任务采用三目标等权代理，不改变能源MPC目标。
            if task in ('economic','carbon','reserve'):weights=np.eye(3)[('economic','carbon','reserve').index(task)]
            benefit=np.maximum(0,weights@values)*self.active
            if benefit.sum()>0:benefit=benefit/(benefit.sum()/self.active.sum())
            priority=self.active*(1+benefit)*(1+np.maximum(0,np.asarray(observation)[18:24]))
            record.update(priority=priority.tolist(),value_proxy=values.tolist(),
                          value_semantics='一步局部误差传播代理，非已证控制效用')
        return record

    def allocate(self, observation, bw, cpu):
        record=self.assess(observation)
        if self.config.coordinator_mode!='schedule':return np.asarray(bw),np.asarray(cpu),record
        if self.service_state is None:raise ValueError('缺少公开队列服务状态')
        s=self.service_state;limit=float(s['deadline_seconds']);B=float(s['bandwidth_bps']);C=float(s['cpu_cycles_per_second'])
        a=np.asarray(s['tx_bits'],dtype=float);q=np.asarray(s['cpu_cycles'],dtype=float)
        n=np.asarray(s['tx_count'],dtype=int)
        if a.shape!=(6,) or q.shape!=(6,) or n.shape!=(6,) or not np.isfinite(np.r_[a,q,B,C,limit]).all() or np.any(a<0) or np.any(q<0) or min(B,C)<0 or limit<=0:raise ValueError('队列服务状态非法')
        fresh=n<s['queue_capacity']
        # 对当前积压及本步最多一个新报文采用串行上界，不读取报文载荷或未来状态。
        A=a+fresh*(s['packet_overhead_bits']+s['payload_bound_bits'])
        D=q+(n+fresh)*s['update_cycles']
        priority=np.asarray(record['priority']);order=sorted(np.flatnonzero(self.active),key=lambda k:(-priority[k],k))
        selected=[]
        def duration(ids):return sum(A[k] for k in ids)/B+sum(D[k] for k in ids)/C if min(B,C)>0 else float('inf')
        for k in order:
            if duration(selected+[k])<=limit:selected.append(int(k))
        b=np.zeros(6);c=np.zeros(6);bound=None
        if selected:
            tx=sum(A[k] for k in selected)/B;cp=sum(D[k] for k in selected)/C
            alpha=tx/(tx+cp);b[selected]=A[selected]/(alpha*limit*B);c[selected]=D[selected]/((1-alpha)*limit*C)
            p=np.zeros(6);p[selected]=priority[selected];p/=p.sum()
            b+=max(0,1-b.sum())*p;c+=max(0,1-c.sum())*p
            bound=float(max(A[k]/(b[k]*B) for k in selected)+max(D[k]/(c[k]*C) for k in selected))
        elif min(B,C)>0:
            # 单个更新也不可保证在周期内完成时继续处理最高优先级积压，显式撤销完成承诺。
            k=order[0];b[k]=c[k]=1.
        record.update(contract='public-queue-service-v1',selected_channels=selected,
                      fresh_admitted=[k for k in selected if fresh[k]],
                      all_active_admitted=len(selected)==int(self.active.sum()),
                      deterministic_completion_bound_seconds=bound,
                      deadline_seconds=limit,service_state=s,
                      service_semantics='控制周期内串行充分条件；须满足载荷上界且无丢包，不保证900秒遥测期限')
        return b,c,record
