"""冻结开发日期上的机制/失效网格；支持MPC及冻结策略，全部失败保留。"""
import os as _os, sys as _sys
if __name__ == '__main__' and not _sys.flags.utf8_mode:
    _os.environ['PYTHONUTF8']='1'
    _os.execv(_sys.executable,[_sys.executable,'-X','utf8',*_sys.argv])

import argparse,copy,csv,hashlib,json,sys,time,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from dataclasses import asdict
from vpp_mappo.config import Config
from vpp_mappo.cyber import CyberSpec
from vpp_mappo.dispatch import DispatchSpec
from vpp_mappo.observed_mpc import ObservedMPC
from vpp_research.real_inputs import connect
from vpp_research.environment import ResearchEnv
from vpp_research.train import load,to_numpy
from vpp_research.ledger import Ledger
from vpp_research.trace import step_record,append
import torch


def grid(group,checkpoint):
    base=[dict(name='nominal')]
    if group=='safety':return base+[dict(name='timeout',fault='timeout'),dict(name='downlink_off',config={'downlink_bps':0.}),dict(name='loss_all',cyber={'packet_loss':1.}),dict(name='resource_shortage',dispatch={'grid_import':100.})]
    if group=='dt':return [dict(name=f'{m}_{s}',dt=m,load_scale=scale) for m in ('none','physics','residual','adaptive') for s,scale in [('nominal',1.),('drift',1.2)]]
    if group=='c3':
        if not checkpoint:raise ValueError('C3消融要求冻结策略检查点')
        return [dict(name=f'{mode}_b{b}_c{c}',resource_mode=mode,cyber={'bandwidth_bps':b,'cpu_cycles_per_second':c}) for mode in ('joint','control','communication','computation') for b in (2000.,10000.) for c in (2e6,1e7)]
    if group=='coordinator':return [dict(name=f'{mode}_{risk}_{task}',config={'coordinator_mode':mode,'risk_adaptive':risk,'task_focus':task}) for mode in ('off','schedule') for risk in (False,True) for task in ('economic','reserve')]
    if group=='network':return [dict(name=n,network=n) for n in ('ieee33','ieee69')]
    raise ValueError(group)


def worker(task):
    a,case,day,seed=task;dest=Path(a.output)/f"{case['name']}_day{day}_seed{seed}";saved=dest/'result.json'
    if saved.exists():return json.loads(saved.read_text())
    if dest.exists() and any(dest.iterdir()):raise ValueError('未完成现场需归档后另起目录：'+str(dest))
    dest.mkdir(parents=True,exist_ok=True)
    c,paths,sets,source=connect(a.data_root,Config.load(a.config),dr_mode=a.dr_mode)
    with zipfile.ZipFile('experiments/v42/dt_study_evidence.zip') as z:
        method=case.get('dt','residual');model=None if method=='none' else json.loads(z.read(method+'.json'))
    agent=None;state=None
    if a.checkpoint:
        state,loaded,agent=load(a.checkpoint)
        # 冻结策略反事实评估，禁止借此声称各消融已重训。
        c=loaded;c.train_csv=str(paths['train']);model=state['dt_model']
        if case.get('dt'):
            with zipfile.ZipFile('experiments/v42/dt_study_evidence.zip') as z:model=None if method=='none' else json.loads(z.read(method+'.json'))
    for k,v in case.get('config',{}).items():setattr(c,k,v)
    if model is None:c.interval_safety=False;c.cyber_dt_mode='hold'
    c._cyber_record={**asdict(CyberSpec.load(getattr(c,'_cyber_record',None) or c.cyber_spec)),**case.get('cyber',{})}
    if case.get('network'):
        c.network_model=case['network'];c.dispatch_spec=f"configs/dispatch_{case['network']}.json"
        c._dispatch_record=DispatchSpec.load(c.dispatch_spec).record()
    if case.get('dispatch'):c._dispatch_record={**DispatchSpec.load(getattr(c,'_dispatch_record',None) or c.dispatch_spec).record(),**case['dispatch']}
    if a.priority_slack is not None:c.coordinator_priority_slack_fraction=a.priority_slack
    c.validate();env=ResearchEnv(c,Path(a.output)/'development.csv',model,reserve_mode='pcc_checked',resource_mode=case.get('resource_mode','joint'))
    if agent and (set(env.data.scenario_names)&set(state['train_scenarios']) or set(env.data.fingerprints)&set(state['train_fingerprints'])):raise ValueError('策略训练评估重叠')
    if model and set(env.data.scenario_names)&set(model['training_scenarios']+model['calibration_scenarios']):raise ValueError('DT训练校准与评估重叠')
    ledger=Ledger(dest/'evaluation.sqlite');record=dict(case=case,day=day,seed=seed,failed=False,steps=0,objective_version=env.objective_version,controller=state['method'] if state else 'mpc',scope='冻结策略反事实' if state else '同信息MPC机制诊断',infos=[],decision_seconds=[])
    (dest/'config.json').write_text(json.dumps(c.__dict__,ensure_ascii=False,indent=2))
    try:
        obs,_=env.reset(seed,day);env.core.profiles['load_kw'][2:]*=case.get('load_scale',1.)
        w=np.array([.2,.3,.5]);env.preference=w;env.reward_mode='weighted'
        planner=ObservedMPC(c,env.spec,env.flex,2);rnn=np.zeros((env.num_agents,1,c.hidden_size),np.float32);masks=np.ones((env.num_agents,1),np.float32)
        for t in range(c.horizon):
            before=obs.copy();start=time.perf_counter();raw=None
            if agent:
                if state['method'] in ('pareto','central'):raw,_,_=agent.act(obs,w,True)
                else:
                    with torch.no_grad():raw,r=agent.policy.act(obs,rnn,masks,deterministic=True)
                    raw=to_numpy(raw);rnn=to_numpy(r)
            else:energy,meta=planner.propose(obs[0])
            record['decision_seconds'].append(time.perf_counter()-start)
            from contextlib import nullcontext
            from unittest.mock import patch
            from vpp_mappo.optimization import DispatchInfeasible
            context=patch.object(env.core,'plan',side_effect=DispatchInfeasible('预注册求解超时注入',reason='timeout')) if case.get('fault')=='timeout' and t in (0,c.horizon-1) else nullcontext()
            ident=ledger.begin('mechanism',0,t)
            try:
                with context:obs,_,_,done,info=env.step(raw) if raw is not None else env.step_candidate(energy,np.ones(6),np.ones(6))
            except Exception as exc:
                ledger.finish(ident,'error',dict(message=str(exc),reason=getattr(exc,'reason',type(exc).__name__),details=getattr(exc,'details',{})));raise
            item=step_record(env,before,raw,w,info,phase='mechanism',episode=0,step=t,attempt_id='mechanism')
            ledger.finish(ident,'success',item);append(dest/'trajectory.jsonl',item)
            record['steps']+=1;record['infos'].append(info)
    except Exception as exc:record.update(failed=True,error=str(exc),reason=getattr(exc,'reason',type(exc).__name__),details=getattr(exc,'details',{}),state_step=env.core.t)
    finally:record['ledger']=ledger.summary();ledger.export();ledger.close()
    infos=record.pop('infos')
    record.update(cost=sum(i['objective_cost']+i['terminal_penalty'] for i in infos),carbon_kg=sum(i['carbon_kg'] for i in infos),reserve_kwh=sum(i['flexibility_kwh'] for i in infos),
        rmse=float(np.sqrt(np.mean([i['research_dt_mse'] for i in infos]))) if infos else None,
        interval_coverage=float(np.mean([i['interval_covered'] for i in infos])) if infos else None,
        guard_certified_steps=sum(i['guard_certificate_survived'] for i in infos),ac_violations=sum(i['ac_violations'] for i in infos),
        emergency_steps=sum(i.get('emergency',False) for i in infos),service_degraded_steps=sum(i.get('emergency_service_degraded',False) for i in infos),
        reserve_invalid=sum(not i['reserve_valid'] for i in infos),terminal_service_failed=infos[-1].get('terminal_service_failed') if infos else None,
        terminal_soc=infos[-1]['ess_soc'] if infos else None,ev_unmet_kwh=sum(i['ev_unmet_kwh'] for i in infos),
        modeled_cyber_energy_j=sum(i['cyber_energy_j'] for i in infos),solver_seconds=sum(i.get('ac_safety_seconds',0.) for i in infos))
    saved.write_text(json.dumps(record,ensure_ascii=False,indent=2));return record


def main(a):
    if not 1<=a.days<=2:raise ValueError('当前冻结开发网格只开放二月27/28两天')
    seeds=[int(v) for v in a.seeds.split(',')]
    if not seeds or min(seeds)<50000 or len(set(seeds))!=len(seeds):raise ValueError('评估种子必须唯一且不小于50000')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    c,paths,sets,source=connect(a.data_root,Config.load(a.config),dr_mode=a.dr_mode)
    dates=sets['validation'].scenario_names[25:27]
    if len(dates)!=2 or any(not d.startswith('2019-02-') for d in dates):raise ValueError('开发日期协议不匹配')
    cases=grid(a.group,a.checkpoint)
    frozen=dict(arguments=vars(a),dates=dates,cases=cases,data=source,config=c.__dict__,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('vpp_mappo','vpp_research') for p in sorted(Path(folder).rglob('*.py'))},runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    frozen=json.loads(json.dumps(frozen));fp=root/'frozen.json'
    if fp.exists() and json.loads(fp.read_text())!=frozen:raise ValueError('冻结协议已变更')
    fp.write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
    with paths['validation'].open() as src,(root/'development.csv').open('w',newline='') as dst:
        r=csv.DictReader(src);w=csv.DictWriter(dst,r.fieldnames);w.writeheader();w.writerows(row for row in r if row['scenario'] in dates)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(worker,[(a,case,day,seed) for case in cases for day in range(a.days) for seed in seeds]))
    result=dict(group=a.group,results=results,total_successful_calls=sum(r['ledger']['successful_calls'] for r in results),failed=sum(r['failed'] for r in results),scope='开发机制诊断，不是最终多种子重训消融；转移到IEEE69的DT区间未重新校准；能耗为模型核算')
    (root/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/v45_ieee33.json');p.add_argument('--data-root',default='data/real');p.add_argument('--checkpoint');p.add_argument('--priority-slack',type=float);p.add_argument('--group',choices=['safety','dt','c3','coordinator','network'],required=True);p.add_argument('--days',type=int,default=1);p.add_argument('--seeds',default='81000');p.add_argument('--workers',type=int,default=2);p.add_argument('--dr-mode',choices=['off','sce-derated'],default='off');p.add_argument('--output',required=True);main(p.parse_args())
