"""公开状态和本控制器发令记录驱动的到达时刻 MPC；预测不冒充执行反馈。"""
import math
import numpy as np
from vpp_mappo.cyber import CyberSpec
from .timing import TimingContract
from .control_v4 import advance_soc
from .safety import guard


def timing(config):
    seconds=config.dt_hours*3600
    if not config.command_timing:return 0,None
    cyber=CyberSpec.load(getattr(config,'_cyber_record',None) or config.cyber_spec)
    contract=TimingContract(downlink_bps=config.downlink_bps,propagation_seconds=config.downlink_propagation_seconds,deadline_seconds=config.command_deadline_steps*seconds)
    result=contract.latency(config.control_cycles/cyber.cpu_cycles_per_second,0) if cyber.cpu_cycles_per_second>0 else dict(delivered=False,deadline_missed=True)
    if not result['delivered'] or result['deadline_missed']:return None,'command_unavailable'
    delay=int(math.ceil(result['seconds']/seconds))
    if delay>config.command_deadline_steps:return None,'expired_before_dispatch_boundary'
    return delay,None


def arrival_observation(planner,observation):
    x=np.asarray(observation,dtype=float).copy();c=planner.config;s=planner.dispatch;f=planner.flex
    if x.shape!=(80,) or not np.isfinite(x).all():raise ValueError('延迟MPC要求80维有限公开观察')
    step=round(x[0]*c.horizon);delay,reason=timing(c)
    if step==0:planner.issued_commands=[]
    history=[q for q in getattr(planner,'issued_commands',[]) if q[1]>=step and step-q[0]<=c.command_deadline_steps]
    planner.issued_commands=history
    meta=dict(mpc_contract='public-80-arrival-mpc-v41',mpc_delay_steps=delay,mpc_prediction='持久状态预测与已发命令；实际安全修正未知',mpc_queue_information_incomplete=False)
    if reason or step+delay>=c.horizon:
        return None,dict(meta,mpc_unavailable=True,mpc_unavailable_reason=reason or 'arrival_after_horizon')
    count=round(x[78]*10)
    if count>len(history):
        # 中途接管时仅能恢复观察中的最新命令，明确记录其余命令未知。
        arrival=step+max(0,int(math.ceil(x[77]-1e-7)))
        history=[(-1,arrival,x[71:77]*1000)]
        meta['mpc_queue_information_incomplete']=count>1
    soc=x[1];low=max(s.soc_min[0],soc-x[54]);high=min(s.soc_max[0],soc+x[54]);used=[]
    for at in range(step,step+delay):
        arrived=[q for q in history if q[1]<=at and (q[0]<0 or at-q[0]<=c.command_deadline_steps)]
        command=max(arrived,key=lambda q:q[0])[2] if arrived else np.zeros(6)
        history=[q for q in history if q[1]>at and (q[0]<0 or at-q[0]<=c.command_deadline_steps)]
        if arrived:used.append(at)
        soc=advance_soc(soc,command[0],s,c.dt_hours);low=advance_soc(low,command[0],s,c.dt_hours);high=advance_soc(high,command[0],s,c.dt_hours)
        x[3]=max(0,x[3]-max(0,command[1])*c.dt_hours/1000)
        x[6]=np.clip(x[6]+command[2]*c.dt_hours/max(1,f.dr_backlog_kwh),0,1)
        x[7]=np.clip(x[7]+max(0,command[2])*c.dt_hours/max(1,f.dr_shift_budget_kwh),0,1)
        x[8]=np.clip(x[8]+max(0,command[3])*c.dt_hours/max(1,f.dr_shed_budget_kwh),0,1)
    x[0]=(step+delay)/c.horizon;x[1]=soc;x[54]=max(soc-low,high-soc);x[5]=max(0,x[5]-delay/c.horizon)
    return x,dict(meta,mpc_unavailable=False,mpc_predicted_soc=float(soc),mpc_predicted_step=step+delay,mpc_consumed_command_boundaries=used)


def propose(planner,observation):
    x,meta=arrival_observation(planner,observation)
    if x is None:return np.zeros(6),dict(meta,mpc_failed=True,mpc_failure_reason=meta['mpc_unavailable_reason'])
    constraints=None
    if planner.config.interval_safety:
        _,audit,certificate=guard(np.zeros(6),x,planner.dispatch,planner.flex,planner.network,planner.config.dt_hours,planner.config.horizon,diagnostics=True,time_limit=planner.config.solver_time_limit)
        constraints=certificate.constraints
        meta.update(mpc_interval_feasible=audit['guard_feasible'],mpc_interval_reason=audit['guard_reason'])
    action,details=planner._propose54(x[:54],constraints)
    step=round(observation[0]*planner.config.horizon)
    if planner.config.command_timing:planner.issued_commands.append((step,step+meta['mpc_delay_steps'],action.copy()))
    return action,dict(details,**meta)
