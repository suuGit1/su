"""在新 PCC 目标下配对检验 DT 估计与 MPC 控制，不筛选有利场景。"""
import argparse
import json
from pathlib import Path
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.observed_mpc import ObservedMPC
from .environment import ResearchEnv, PCC_OBJECTIVE_VERSION
from .suite import stats


def run(config, dt_folder, output, seeds=(1910,1911,1912)):
    models={m:json.loads((Path(dt_folder)/(m+'.json')).read_text()) for m in ('physics','residual')}
    rows=[];result=dict(objective_version=PCC_OBJECTIVE_VERSION,network_model=config.network_model,seeds=list(seeds),rows=rows,
        limits=['合成场景配对功能实验，不代表真实数据显著性','区间覆盖为所有9个量同时覆盖的逐步比例；不是完整场景覆盖'])
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        for method in models:
            record=dict(seed=seed,method=method,failed=False);rows.append(record)
            try:
                env=ResearchEnv(config,dt_model=models[method],reserve_mode='pcc_checked');obs,_=env.reset(seed)
                mpc=ObservedMPC(config,env.spec,env.flex,2);infos=[]
                for t in range(config.horizon):
                    action,_=mpc.propose(obs[0,:54]);obs,_,_,_,info=env.step_candidate(action,np.ones(6),np.ones(6));infos.append(info)
                record.update(cost=sum(i['objective_cost']+i['terminal_penalty'] for i in infos),
                    rmse=float(np.sqrt(np.mean([i['research_dt_mse'] for i in infos]))),
                    step_coverage=float(np.mean([i['interval_covered'] for i in infos])),
                    ac_violations=sum(i['ac_violations'] for i in infos),
                    service_unmet_kwh=sum(i['ev_unmet_kwh'] for i in infos),terminal_dr_kwh=infos[-1]['dr_backlog_kwh'])
            except (RuntimeError,ValueError) as exc:record.update(failed=True,error=str(exc))
            path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(seed,method,record,flush=True)
    deltas=[]
    for seed in seeds:
        pair=[r for r in rows if r['seed']==seed]
        if not any(r['failed'] for r in pair):deltas.append(pair[1]['cost']-pair[0]['cost'])
    result['paired_cost_residual_minus_physics']=stats(deltas) if deltas else None
    result['failed_pairs']=len(seeds)-len(deltas)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dt-folder',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(Config.load('configs/research_smoke.json'),a.dt_folder,a.output)
