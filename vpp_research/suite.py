"""统一交互预算、多训练种子、验证前沿和独立测试偏好的实验驱动。"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from vpp_mappo.pareto import non_dominated,hypervolume,igd
from .train import train,evaluate

VALIDATION_WEIGHTS=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1/3,1/3,1/3]]
TEST_WEIGHTS=[[.2,.3,.5],[.6,.1,.3],[.1,.7,.2]]
FIXED_WEIGHTS=[[1.,0.,0.],[.2,.5,.3],[.3,.2,.5]]
REFERENCE=[-50.,-50.,0.]


def feasible(r):
    return not r['failed'] and r['violations']==0 and r['ac_violations']==0 and r['ac_failed']==0 and r['reserve_invalid']==0 and r['ev_unmet_kwh']<1e-5 and abs(r['dr_backlog_kwh'])<1e-5


def points(rows,expected):
    groups={}
    for r in rows:groups.setdefault(tuple(r['preference']),[]).append(r)
    return [np.mean([r['vector'] for r in rs],axis=0).tolist() for rs in groups.values() if len(rs)==expected and all(feasible(r) for r in rs)]


def stats(values):
    a=np.asarray(values,dtype=float);n=len(a);mean=float(a.mean())
    se=float(a.std(ddof=1)/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=mean,std=float(a.std(ddof=1)) if n>1 else None,
                ci95=[mean-student_t.ppf(.975,n-1)*se,mean+student_t.ppf(.975,n-1)*se] if n>1 else None)


def run(config,dt_folder,output,seeds=(1,2,3,4,5),episodes=12,families=None):
    if episodes<3 or episodes%3:raise ValueError('预算需为至少 3 的 3 倍数，以均分固定权重模型族预算')
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('实验目录非空')
    out.mkdir(parents=True,exist_ok=True)
    models={k:json.loads((Path(dt_folder)/(k+'.json')).read_text()) for k in ('hold','physics','residual')}
    all_results=[];per_seed=[];checkpoints={}
    for seed in seeds:
        entries=[]
        for family in (families or ('ordinary','fixed','pareto','pareto_no_residual','pareto_no_coordinator')):
            cfg=replace(config,seed=seed,episodes=episodes,coordinator_mode='off' if family=='pareto_no_coordinator' else config.coordinator_mode)
            model=models['physics' if family=='pareto_no_residual' else 'residual']
            val=[];test=[];members=FIXED_WEIGHTS if family=='fixed' else [[1.,0.,0.]]
            for index,w in enumerate(members):
                method=family if family in ('ordinary','fixed') else 'pareto'
                cfg.episodes=episodes//len(members);folder=out/f'seed_{seed}'/f'{family}_{index}'
                train(cfg,model,folder,method,w,scenario_offset=index*cfg.episodes)
                checkpoint=folder/'latest.pt';checkpoints[(seed,family)]=checkpoint
                if family=='fixed':
                    # 每个固定权重模型提供一个点，标记其训练偏好而不是复制为多项偏好。
                    vr=evaluate(checkpoint,[w],[8000,8001]);tr=evaluate(checkpoint,[w],[9000,9001,9002])
                else:
                    vr=evaluate(checkpoint,VALIDATION_WEIGHTS if method=='pareto' else [[1.,0.,0.]],[8000,8001])
                    tr=evaluate(checkpoint,TEST_WEIGHTS if method=='pareto' else [[1.,0.,0.]],[9000,9001,9002])
                val+=vr;test+=tr
            entry=dict(seed=seed,family=family,training_env_steps=episodes*config.horizon,validation=val,test=test,
                       validation_points=points(val,2),test_points=points(test,3))
            entries.append(entry);all_results.append(entry)
            print(f'seed={seed} family={family} steps={episodes*config.horizon}',flush=True)
        # IGD 参考集只由验证场景构造；不是已知真实 Pareto 前沿。
        union=[p for e in entries for p in e['validation_points']]
        reference=[union[i] for i in non_dominated(union)] if union else []
        for e in entries:
            p=e['test_points'];e['hv']=hypervolume(p,REFERENCE)
            e['igd_validation_reference']=igd(p,reference) if p and reference else None
            e['evaluation_failures']=sum(r['failed'] for r in e['test']);e['empirical_reference_front']=reference
        per_seed.extend([{k:e[k] for k in ('seed','family','training_env_steps','hv','igd_validation_reference','evaluation_failures')} for e in entries])
    aggregate={}
    for family in sorted({e['family'] for e in per_seed}):
        rows=[e for e in per_seed if e['family']==family]
        igds=[e['igd_validation_reference'] for e in rows if e['igd_validation_reference'] is not None]
        aggregate[family]=dict(hv=stats([e['hv'] for e in rows]),igd=stats(igds) if igds else None,
            undefined_igd_runs=len(rows)-len(igds),failed_episodes=sum(e['evaluation_failures'] for e in rows))
    if all(any(e['family']=='ordinary' and e['seed']==seed for e in per_seed) for seed in seeds):
        for family in aggregate:
            deltas=[next(e['hv'] for e in per_seed if e['seed']==seed and e['family']==family)-next(e['hv'] for e in per_seed if e['seed']==seed and e['family']=='ordinary') for seed in seeds]
            aggregate[family]['paired_hv_delta_vs_ordinary']=stats(deltas)
    # 相同训练策略下，评估阶段切换区间保护，单独标记分布变化，不冒充重训练消融。
    stress=[]
    for seed in seeds:
        checkpoint=checkpoints[(seed,'pareto')]
        for name,params in [('nominal',None),('loss_30pct',{'packet_loss':.3}),('cpu_outage',{'cpu_cycles_per_second':0.}),('bandwidth_10pct',{'bandwidth_bps':1000.}),('wind_jump',{'wind_scale':2.}),('load_shock',{'load_scale':1.4})]:
            for robust in (False,True):
                results=evaluate(checkpoint,[[.2,.3,.5]],[9500,9501],robust=robust,stress=params)
                stress.append(dict(seed=seed,scenario=name,interval_guard=robust,results=results))
    from .environment import ResearchEnv
    from vpp_mappo.observed_mpc import ObservedMPC
    mpc_reference=[]
    for scenario in (9000,9001,9002):
        env=ResearchEnv(config,dt_model=models['residual']);obs,_=env.reset(scenario)
        planner=ObservedMPC(config,env.spec,env.flex,2);infos=[];failures=0
        try:
            for _ in range(config.horizon):
                action,meta=planner.propose(obs[0,:54]);failures+=meta['mpc_failed']
                obs,_,_,_,info=env.step_candidate(action,np.ones(6),np.ones(6));infos.append(info)
            mpc_reference.append(dict(seed=scenario,vector=np.sum([i['objective_vector'] for i in infos],axis=0).tolist(),
                mpc_failures=failures,violations=sum(i['constraint_violations'] for i in infos),ac_violations=sum(i['ac_violations'] for i in infos)))
        except (RuntimeError,ValueError) as exc:mpc_reference.append(dict(seed=scenario,failed=True,error=str(exc)))
    result=dict(protocol='five-seed-budget-matched-smoke-v1',cyber_mode=config.cyber_mode,seeds=list(seeds),episodes_per_family=episodes,
        mpc_economic_reference=mpc_reference,
        common_training_interactions=episodes*config.horizon,reference_point=REFERENCE,validation_weights=VALIDATION_WEIGHTS,
        unseen_test_weights=TEST_WEIGHTS,aggregate=aggregate,per_seed=per_seed,raw_results=all_results,stress=stress,
        limitations=['交互预算相同，不声称训练耗时/参数数量完全相同','IGD 使用验证集经验前沿，不是真实 Pareto 前沿','短程实验不证明收敛','区间保护只对所建模盒内单步线性约束检查，不是闭环 AC 鲁棒保证'])
    result=finalize(result)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def finalize(result):
    """只用全部训练种子的验证点冻结共同 IGD 参考集，不读取测试点选参考。"""
    union=[p for e in result['raw_results'] for p in e['validation_points']]
    reference=[union[i] for i in non_dominated(union)] if union else []
    per_seed=[]
    for e in result['raw_results']:
        e['empirical_reference_front']=reference
        e['igd_validation_reference']=igd(e['test_points'],reference) if e['test_points'] and reference else None
        per_seed.append({k:e[k] for k in ('seed','family','training_env_steps','hv','igd_validation_reference','evaluation_failures')})
    aggregate={}
    for family in sorted({e['family'] for e in per_seed}):
        rows=[e for e in per_seed if e['family']==family];igds=[e['igd_validation_reference'] for e in rows if e['igd_validation_reference'] is not None]
        item=dict(hv=stats([e['hv'] for e in rows]),igd=stats(igds) if igds else None,
            undefined_igd_runs=len(rows)-len(igds),failed_episodes=sum(e['evaluation_failures'] for e in rows))
        if all(any(e['family']=='ordinary' and e['seed']==r['seed'] for e in per_seed) for r in rows):
            delta=[r['hv']-next(e['hv'] for e in per_seed if e['family']=='ordinary' and e['seed']==r['seed']) for r in rows]
            item['paired_hv_delta_vs_ordinary']=stats(delta)
        aggregate[family]=item
    result.update(aggregate=aggregate,per_seed=per_seed,igd_reference_scope='common_validation_union_all_training_seeds',igd_reference_front=reference)
    return result
