"""固定检查点的日内偏好切换；环境任务轨迹相同，仅改变actor所见偏好。"""
import os as _os, sys as _sys
if __name__ == '__main__' and not _sys.flags.utf8_mode:
    _os.environ['PYTHONUTF8']='1'
    _os.execv(_sys.executable,[_sys.executable,'-X','utf8',*_sys.argv])

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from vpp_research.train import load,to_numpy
from vpp_research.environment import ResearchEnv


def run(a):
    state,c,agent=load(a.checkpoint)
    schedule=json.loads(Path(a.schedule).read_text())
    steps=[r['step'] for r in schedule]
    if steps!=sorted(set(steps)) or not steps or steps[0]!=0 or any(type(t)!=int or t<0 or t>=c.horizon for t in steps):raise ValueError('切换步须唯一递增、从0开始且在回合内')
    for row in schedule:
        w=np.asarray(row['preference'],dtype=float)
        if w.shape!=(3,) or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1):raise ValueError('切换偏好非法')
    env=ResearchEnv(c,a.csv,state['dt_model'],state['robust'],ac_safe=True,resource_mode=state.get('resource_mode','joint'),reserve_mode=state['reserve_mode'])
    if env.data is None or not 1<=a.days<=len(env.data.profiles):raise ValueError('必须使用足量真实完整日期')
    if any(not (d.startswith('2019-03-') or d in ('2019-02-27','2019-02-28')) for d in env.data.scenario_names):raise ValueError('本开发入口不允许解封确认日期')
    reserved=set(state['train_scenarios'])|set(state['dt_model']['training_scenarios'])|set(state['dt_model']['calibration_scenarios'])
    if set(env.data.scenario_names)&reserved or set(env.data.fingerprints)&set(state['train_fingerprints']):raise ValueError('数据重叠')
    if any(a.seed+i in state['train_seeds'] for i in range(a.days)):raise ValueError('训练评估种子重叠')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    frozen=dict(checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),csv_sha256=hashlib.sha256(Path(a.csv).read_bytes()).hexdigest(),
                schedule=schedule,seed=a.seed,days=a.days,method=state['method'],objective_version=state['objective_version'],
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('vpp_research','vpp_mappo') for p in sorted(Path(folder).rglob('*.py'))})
    path=root/'frozen.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('冻结配置已变更')
    path.write_text(json.dumps(frozen,indent=2))
    from vpp_research.ledger import Ledger
    from vpp_research.trace import step_record
    results=[]
    for index in range(a.days):
        for mode in ('switch','fixed_initial','fixed_balanced'):
            out=root/f'{index:03d}_{mode}.json'
            if out.exists():results.append(json.loads(out.read_text()));continue
            target=root/f'{index:03d}_{mode}'
            if target.exists() and any(target.iterdir()):raise ValueError('不完整切换现场必须先归档，不得静默覆盖')
            ledger=Ledger(target/'evaluation.sqlite')
            result=dict(date=env.data.scenario_names[index],seed=a.seed+index,mode=mode,failed=False,steps=[])
            obs,_=env.reset(a.seed+index,index);rnn=np.zeros((env.num_agents,1,c.hidden_size),np.float32);masks=np.ones((env.num_agents,1),np.float32)
            try:
                for t in range(c.horizon):
                    requested=np.asarray([r['preference'] for r in schedule if r['step']<=t][-1])
                    actor_w=requested if mode=='switch' else np.asarray(schedule[0]['preference']) if mode=='fixed_initial' else np.ones(3)/3
                    env.preference=requested.copy();env.reward_mode='weighted'
                    started=time.perf_counter()
                    if state['method'] in ('pareto','central'):action,_,_=agent.act(obs,actor_w,True)
                    else:
                        with torch.no_grad():action,new_rnn=agent.policy.act(obs,rnn,masks,deterministic=True)
                        action=to_numpy(action);rnn=to_numpy(new_rnn)
                    inference=time.perf_counter()-started
                    before=obs.copy();ident=ledger.begin('switch',0,t)
                    try:obs,_,_,_,info=env.step(action)
                    except Exception as exc:
                        ledger.finish(ident,'error',dict(error=str(exc)));raise
                    ledger.finish(ident,'success',step_record(env,before,action,requested,info,phase='switch',step=t,episode=0,attempt_id='switch'))
                    vector=np.asarray(info['objective_vector'])
                    result['steps'].append(dict(step=t,requested=requested.tolist(),actor_preference=actor_w.tolist(),
                        action=np.asarray(action).reshape(-1).tolist(),vector=vector.tolist(),utility=float(requested@vector),
                        inference_seconds=inference,ac_violations=info['ac_violations'],reserve_valid=bool(info['reserve_valid']),
                        terminal_service_failed=info.get('terminal_service_failed',False),soc=info['ess_soc'],certificate=bool(info['guard_certificate_survived']),intervention=info['shield_l1_kw'],
                        executed_action=[float(info[k]) for k in ('ess_power_kw','ev_charge_kw','dr_shift_kw','dr_shed_kw','pv_curtail_kw','wind_curtail_kw')]))
                result['completed']=len(result['steps'])==c.horizon
                result['utility']=sum(s['utility'] for s in result['steps'])
                result['terminal_service_failed']=info.get('terminal_service_failed',False)
                result['terminal_soc']=info['ess_soc']
            except (RuntimeError,ValueError) as exc:result.update(failed=True,error=str(exc),state_step=env.core.t,completed=False)
            result['ledger']=ledger.summary();ledger.export();ledger.close()
            out.write_text(json.dumps(result,ensure_ascii=False,indent=2));results.append(result)
    (root/'results.json').write_text(json.dumps(dict(cases=results,scope='开发检查点切换诊断；不是动态最优性或收敛证明'),ensure_ascii=False,indent=2))
    print(json.dumps(dict(cases=len(results),failed=sum(r['failed'] for r in results)),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--csv',required=True);p.add_argument('--schedule',default='configs/preference_switch.json');p.add_argument('--days',type=int,default=1);p.add_argument('--seed',type=int,default=81000);p.add_argument('--output',default='runs/preference_switch');run(p.parse_args())
