"""指定净购电目标的备用激活：完整执行克隆轨迹，保留 AC 不可行与服务损失。"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.observed_mpc import ObservedMPC
from vpp_mappo.flex_resources import grid_power
from .environment import ResearchEnv


def target_plan(core,target):
    original=core.network.spec;smax=core.network.smax
    for margin in (0.,.002,.004,.006):
        try:
            core.network.spec=replace(original,voltage_min=original.voltage_min+margin,voltage_max=original.voltage_max-margin)
            core.network.smax=smax*(1-5*margin)
            plans,meta=core.plan(grid_target=target)
        except RuntimeError:continue
        finally:core.network.spec=original;core.network.smax=smax
        audit=core.network.audit(core.row(),plans[0]['action'])
        if audit['ac_converged'] and audit['ac_violations']==0:return plans[0],meta
    raise RuntimeError('指定备用激活目标未找到通过 AC 验收的动作')


def activate(core,baseline,reserve_kw,direction,duration_steps=1):
    # 独立克隆防止上/下调试验改变主轨迹；求解仅使用当前时刻和已接入车辆。
    e=copy.deepcopy(core);target=baseline+direction*reserve_kw;rows=[]
    result=dict(direction=direction,requested_kw=reserve_kw,duration_steps=duration_steps,failed=False)
    try:
        for _ in range(duration_steps):
            plan,meta=target_plan(e,target);*_,done,info=e.step_physical(plan)
            rows.append(dict(grid_kw=info['grid_power_kw'],target_kw=target,tracking_error_kw=abs(info['grid_power_kw']-target),
                ac_violations=info['ac_violations'],ev_unmet_kwh=info['ev_unmet_kwh'],dr_backlog_kwh=info['dr_backlog_kwh']))
        # 激活结束后继续执行当前信息 MPC，检查后续任务是否仍可完成。
        while e.t<e.config.horizon:
            plan,_=e.plan();*_,done,info=e.step_physical(plan[0])
            if info['constraint_violations'] or info['ac_violations']:raise RuntimeError('激活后续轨迹出现约束违规')
        result.update(ev_terminal_unmet_kwh=sum(max(v,0) for v in e.remaining.values()),dr_terminal_backlog_kwh=e.backlog)
    except (RuntimeError,ValueError) as exc:result.update(failed=True,error=str(exc))
    result['activation_steps']=rows;return result


def run(config,model,output,seeds=range(1700,1708)):
    rows=[]
    for seed in seeds:
        env=ResearchEnv(config,dt_model=model,ac_safe=False);obs,_=env.reset(seed);planner=ObservedMPC(config,env.spec,env.flex,2)
        for t in range(config.horizon):
            old=env.core.row().copy();a,_=planner.propose(obs[0,:54]);obs,_,_,done,info=env.step_candidate(a,np.ones(6),np.ones(6))
            if done:continue
            kw=info['flexibility_kwh']/config.dt_hours
            for direction in (-1,1):
                r=activate(env.core,info['grid_power_kw'],kw,direction);r.update(seed=seed,t=t,declared_kw=kw);rows.append(r)
    out=Path(output);out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(dict(scope='旧线性备用的实际下一步激活及后续回补验证；失败全部保留',rows=rows),ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dt-model',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Config.load('configs/research_smoke.json'),json.loads(Path(a.dt_model).read_text()),a.output)
