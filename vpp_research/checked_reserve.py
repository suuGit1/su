"""保守的当前状态 AC 端点备用校核；不改变旧目标版本，不承诺未来扰动递归可行。"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .reserve_activation import target_plan


def checked_capacity(core,baseline,upper_kw,iterations=8):
    """固定基线的两端都通过同约束 MILP 和 AC 检查；有限分辨率向下取保守值。"""
    if not np.isfinite([baseline,upper_kw]).all() or upper_kw<0:raise ValueError('备用基线/上界非法')
    def valid(q):
        try:
            for sign in (-1,1):target_plan(core,baseline+sign*q)
            return True
        except RuntimeError:return False
    if not valid(0.):return dict(kw=0.,baseline_feasible=False,endpoint_checked=False)
    if valid(upper_kw):return dict(kw=float(upper_kw),baseline_feasible=True,endpoint_checked=True)
    low=0.;high=float(upper_kw)
    for _ in range(iterations):
        mid=(low+high)/2
        if valid(mid):low=mid
        else:high=mid
    return dict(kw=low,baseline_feasible=True,endpoint_checked=True)


def ac_economic_plan(core):
    original=core.network.spec;smax=core.network.smax
    for margin in (0.,.002,.004,.006):
        try:
            core.network.spec=replace(original,voltage_min=original.voltage_min+margin,voltage_max=original.voltage_max-margin)
            core.network.smax=smax*(1-margin*5);p,_=core.plan()
        except RuntimeError:continue
        finally:core.network.spec=original;core.network.smax=smax
        audit=core.network.audit(core.row(),p[0]['action'])
        if audit['ac_converged'] and audit['ac_violations']==0:return p[0]
    raise RuntimeError('后续 AC 经济规划无已验证可行动作')


def replay(core,baseline,q,direction):
    e=copy.deepcopy(core);r=dict(failed=False)
    try:
        p,_=target_plan(e,baseline+direction*q);*_,info=e.step_physical(p)
        r['tracking_error_kw']=abs(info['grid_power_kw']-baseline-direction*q)
        while e.t<e.config.horizon:
            p=ac_economic_plan(e);*_,info=e.step_physical(p)
            if info['constraint_violations'] or info['ac_violations']:raise RuntimeError('激活后约束违规')
        r.update(ev_unmet_kwh=sum(max(0,v) for v in e.remaining.values()),dr_backlog_kwh=e.backlog)
        if r['ev_unmet_kwh']>1e-5 or abs(e.backlog)>1e-5:raise RuntimeError('激活后的服务任务未完成')
    except RuntimeError as exc:r.update(failed=True,error=str(exc))
    return r


def run(config,model,output):
    from .environment import ResearchEnv
    from vpp_mappo.observed_mpc import ObservedMPC
    rows=[]
    for seed in range(1700,1708):
        e=ResearchEnv(config,dt_model=model);obs,_=e.reset(seed);mpc=ObservedMPC(config,e.spec,e.flex,2)
        for t in range(config.horizon):
            a,_=mpc.propose(obs[0,:54]);obs,_,_,done,info=e.step_candidate(a,np.ones(6),np.ones(6))
            if done:continue
            old=info['flexibility_kwh']/config.dt_hours;g=info['grid_power_kw'];capacity=checked_capacity(e.core,g,old)
            record=dict(seed=seed,t=t,old_kw=old,checked=capacity,activations=[])
            if capacity['endpoint_checked'] and capacity['kw']>1e-5:
                record['activations']=[dict(direction=s,**replay(e.core,g,capacity['kw'],s)) for s in (-1,1)]
            rows.append(record)
    result=dict(protocol='current-state-ac-endpoint-reserve-audit-v1',rows=rows,
        limits=['使用当前真实状态进行现场备用确认，不给策略添加真值输入','改变了备用能力估计方式；旧策略没有按此目标重训练','只检查两个端点和所测后续轨迹；非全区间/所有未来扰动安全证明','无法维持基线或无正容量时保留记录，不算成功履约'])
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    from vpp_mappo.config import Config
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Config.load('configs/research_smoke.json'),json.loads(Path(a.model).read_text()),a.output)
