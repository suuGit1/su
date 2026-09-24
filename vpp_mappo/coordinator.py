"""规则式任务/风险协调器；输入仅为公开观察，风险评分不是违约概率。"""
import numpy as np
from .cyber import CHANNELS

CONTRACT_VERSION='observation-task-risk-v1'


class TaskRiskCoordinator:
    def __init__(self,config,dispatch,flex):
        self.config=config;self.dispatch=dispatch;self.flex=flex

    def assess(self,observation):
        x=np.asarray(observation,dtype=float)
        if x.shape!=(54,) or not np.isfinite(x).all():raise ValueError('协调器要求有限的 54 维公开观察')
        c=self.config;step=round(x[0]*c.horizon);left=max(0,c.horizon-step)
        ages=x[18:24];stale=ages>c.coordinator_aoi_limit_steps
        scores=np.maximum(0,ages/c.coordinator_aoi_limit_steps-1)
        reasons=[['stale_telemetry'] if flag else [] for flag in stale]
        need=max(0,x[3]*1000);deadline=max(0,round(x[5]*c.horizon))
        # 观察只有聚合 EV 信息，因此这里用站级能力估计乐观服务余量。
        ev_slack=deadline-need/(self.flex.ev_station_kw*c.dt_hours)
        if need>1e-7 and ev_slack<c.coordinator_deadline_margin_steps:
            scores[1]+=1;reasons[1].append('ev_deadline_pressure')
        backlog=max(0,x[6]*max(1,self.flex.dr_backlog_kwh))
        repay_steps=backlog/max(1e-12,self.flex.dr_repay_kw*c.dt_hours)
        if backlog>1e-7 and left-repay_steps<c.coordinator_deadline_margin_steps:
            scores[2]+=1;reasons[2].append('dr_terminal_pressure')
        soc_margin=min(x[1]-self.dispatch.soc_min[0],self.dispatch.soc_max[0]-x[1])
        if soc_margin<c.coordinator_soc_margin:
            scores[0]+=1;reasons[0].append('soc_near_limit')
        # 每个资源始终保留基础权重，防止按风险排队导致其它资源永久饥饿。
        task=self.config.task_focus
        fraction=self.config.coordinator_reserved_fraction
        uncertainty=np.zeros(6)
        if self.config.risk_adaptive:
            widths=np.asarray(getattr(self,'interval_width',np.zeros(9)))
            uncertainty=np.minimum(5.,widths[[0,1,2,4,7,8]]/.05)
            scores+=uncertainty
            preference=np.asarray(getattr(self,'preference',[1.,0.,0.]))
            task_weights=np.array([0.,preference[0],0.,0.,preference[1],preference[1]])
            if task=='carbon':task_weights[4:]+=1.
            elif task=='reserve':task_weights[[0,1,2]]+=1.
            elif task=='economic':task_weights[[0,1]]+=1.
            scores+=task_weights
            fraction=min(.95,fraction+(1-fraction)*float(scores.max())/(1+float(scores.max())))
        priority=1+scores
        tasks=[dict(kind='refresh_dt',channel=name,age_steps=float(ages[k]),
                    due_in_steps=float(c.coordinator_aoi_limit_steps-ages[k]),reasons=reasons[k]) for k,name in enumerate(CHANNELS)]
        if need>1e-7:tasks.append(dict(kind='ev_service',remaining_kwh=need,due_in_steps=deadline,station_slack_steps=float(ev_slack)))
        if backlog>1e-7:tasks.append(dict(kind='dr_repay',remaining_kwh=backlog,due_in_steps=left))
        return dict(contract=CONTRACT_VERSION,step=step,risk_channels=int(np.count_nonzero(scores)),
                    risk_scores=scores.tolist(),priority=priority.tolist(),tasks=tasks,
                    risk_semantics='heuristic_thresholds_not_probability',task_focus=task,
                    interval_risk=uncertainty.tolist(),reserved_fraction=float(fraction))

    def allocate(self,observation,bw,cpu):
        record=self.assess(observation)
        if self.config.coordinator_mode!='schedule':return np.asarray(bw),np.asarray(cpu),record
        p=np.asarray(record['priority']);p/=p.sum();fraction=record['reserved_fraction']
        def blend(values):
            v=np.asarray(values,dtype=float)
            return (1-fraction)*v/max(1,float(v.sum()))+fraction*p
        return blend(bw),blend(cpu),record
