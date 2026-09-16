"""独立场景的 DT 估计验证与相同 MPC 下的控制验证。"""
from dataclasses import replace
import copy
import json
from pathlib import Path
import numpy as np
from vpp_mappo.observed_mpc import ObservedMPC
from .environment import ResearchEnv
from .dt import TARGETS,fit,assess,save


def collect(config,seeds,mode,csv_path=None):
    records=[]
    for index,seed in enumerate(seeds):
        c=copy.copy(config);c.cyber_dt_mode=mode;e=ResearchEnv(c,csv_path,vector_metrics=False)
        if e.data and len(seeds)>len(e.data.profiles):raise ValueError('真实独立场景数量不足')
        e.reset(seed,index)
        rng=np.random.default_rng(seed+71)
        for t in range(c.horizon):
            x=e.raw_observation();y=e.features(e.sensor_payloads())[TARGETS]
            records.append(dict(scenario=e.data.scenario_names[index] if e.data else f'synthetic:{seed}',step=t,x=x.tolist(),y=y.tolist()))
            # 同一候选命令序列用于所有 DT 基线，保证估计比较共用物理轨迹。
            action=np.array([rng.uniform(-100,100),rng.uniform(0,20),rng.uniform(-20,20),rng.uniform(0,10),0,0])
            e.step_candidate(action,np.ones(6),np.ones(6))
    return records


def control(config,model,seeds,robust=False,csv_path=None):
    rows=[]
    for index,seed in enumerate(seeds):
        e=ResearchEnv(config,csv_path,dt_model=model,robust=robust,vector_metrics=False);obs,_=e.reset(seed,index)
        planner=ObservedMPC(config,e.spec,e.flex,2);cost=0.;violations=interventions=failures=0;error=[];coverage=[]
        for _ in range(config.horizon):
            truth=e.features(e.sensor_payloads())[TARGETS]
            error.append(float(np.mean((obs[0,TARGETS]-truth)**2)))
            coverage.append(bool(np.all(np.abs(obs[0,TARGETS]-truth)<=np.array(model['halfwidth'])+1e-6)))
            a,m=planner.propose(obs[0,:54]);obs,_,_,_,i=e.step_candidate(a,np.ones(6),np.ones(6))
            cost+=i['cost']+i['terminal_penalty'];violations+=i['constraint_violations'];interventions+=i['shield_l1_kw']>1e-5;failures+=m['mpc_failed']
        rows.append(dict(seed=seed,objective=cost,violations=violations,safety_interventions=int(interventions),mpc_failures=failures,
            state_rmse=float(np.sqrt(np.mean(error))),scenario_covered=all(coverage)))
    return rows


def run(config,output):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('DT 输出目录非空')
    out.mkdir(parents=True,exist_ok=True)
    splits=dict(train=list(range(1000,1024)),calibration=list(range(1100,1112)),test=list(range(1200,1208)))
    results={}
    datasets={mode:{split:collect(config,seeds,mode) for split,seeds in splits.items()} for mode in ('hold','physics')}
    for method in ('hold','physics','residual'):
        d=datasets['hold' if method=='hold' else 'physics']
        model=fit(d['train'],d['calibration'],method=method)
        save(model,out/(method+'.json'))
        results[method]=dict(estimation=assess(model,d['test']),closed_loop=control(config,model,splits['test']))
    results['splits']=splits;results['selection']='residual 是预注册研究分支；即使劣于基线也不隐瞒，未按测试结果调参'
    results['coverage_scope']='按独立场景块校准；策略变化/分布偏移时仅报告实测覆盖，不承诺名义保证'
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    # 保留可审计的观测与离线真值标签；这些标签不进入在线观察。
    (out/'datasets.json').write_text(json.dumps(datasets),encoding='utf-8')
    return results
