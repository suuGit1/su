"""仅利用公开 DT 和本地命令记录解码动作；估计边界不是安全证书。"""
import numpy as np


def queue_features(env):
    pending=env.command_queue.pending if env.command_queue else []
    latest=max(pending,key=lambda p:p[1]) if pending else None
    command=latest[2] if latest else np.zeros(6)
    eta=max(0,latest[0]-env.pipeline.now)/(env.config.dt_hours*3600) if latest else 0.
    # 不把现场执行功率或真实SOC通过零时延旁路输入策略。
    return np.r_[env.last_received_command/1000,command/1000,eta,len(pending)/10,float(env.config.command_timing)]


def advance_soc(soc,power,spec,hours):
    delta=(-power/spec.efficiency if power>=0 else -power*spec.efficiency)*hours/spec.capacities[0]
    return np.clip(soc+delta,spec.soc_min[0],spec.soc_max[0])


def ess_bounds(env,obs):
    spec=env.spec;dt=env.config.dt_hours
    lo=np.clip(obs[1]-obs[54],spec.soc_min[0],spec.soc_max[0])
    hi=np.clip(obs[1]+obs[54],spec.soc_min[0],spec.soc_max[0])
    # 预测新指令生效前的已知命令，队列消费规则与执行模块一致。
    if env.command_queue:
        now=env.pipeline.now;seconds=dt*3600;cap=env.cyber_spec.cpu_cycles_per_second
        timing=env.timing_contract.latency(env.control_cycles/cap,0) if cap>0 else {'seconds':None}
        delay=timing['seconds']
        if delay is not None:
            pending=list(env.command_queue.pending)
            for boundary in range(min(env.config.horizon,max(0,int(np.ceil(delay/seconds))))):
                at=now+boundary*seconds
                arrived=[p for p in pending if p[0]<=at and at-p[1]<=env.timing_contract.deadline_seconds]
                power=max(arrived,key=lambda p:p[1])[2][0] if arrived else 0.
                pending=[p for p in pending if p[0]>at and at-p[1]<=env.timing_contract.deadline_seconds]
                lo=advance_soc(lo,power,spec,dt);hi=advance_soc(hi,power,spec,dt)
    discharge=min(spec.power_max[0],max(0,lo-spec.soc_min[0])*spec.capacities[0]*spec.efficiency/dt)
    charge=min(spec.power_max[0],max(0,spec.soc_max[0]-hi)*spec.capacities[0]/spec.efficiency/dt)
    return -charge,discharge


def decode(env,raw):
    obs=env.encode()[0][0];x=np.tanh(raw[:,0]);low,high=ess_bounds(env,obs)
    if not env.config.v4_action_bounds:low,high=-env.spec.power_max[0],env.spec.power_max[0]
    ess=x[0]*(high if x[0]>=0 else -low)
    # 所有学习基线使用相同映射，零均值动作从低弃电起点探索。
    fraction=(np.tanh(raw[:,0]-env.config.v4_curtail_bias)+1)/2
    energy=np.array([ess,(x[1]+1)*env.flex.ev_station_kw/2,
        x[2]*(env.flex.dr_shift_kw if x[2]>=0 else env.flex.dr_repay_kw),fraction[3]*env.flex.dr_shed_kw,
        fraction[4]*max(0,obs[11]*1000),fraction[5]*max(0,obs[12]*1000)])
    bw=np.logaddexp(0,raw[6:12,0]) if env.num_agents==18 else np.ones(6)
    cpu=np.logaddexp(0,raw[12:18,0]) if env.num_agents==18 else np.ones(6)
    if env.resource_mode in ('control','computation'):bw=np.ones(6)
    if env.resource_mode in ('control','communication'):cpu=np.ones(6)
    return energy,bw,cpu
