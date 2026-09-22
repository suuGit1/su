"""预注册扰动网格：同信息 MPC 的 DT 控制作用与备用端点核验。"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.cyber import CyberSpec
from vpp_mappo.observed_mpc import ObservedMPC
from vpp_mappo.flex_resources import grid_power
from .environment import ResearchEnv
from .dt import TARGETS
from .suite import stats

SCENARIOS={
    'nominal':{},
    'uplink_slow':{'bandwidth_bps':2000.},
    'compute_slow':{'cpu_cycles_per_second':2.e6},
    'packet_loss':{'packet_loss':.3},
    'load_up':{'load_scale':1.2},
    'wind_up':{'wind_scale':1.5},
    'combined':{'bandwidth_bps':2000.,'packet_loss':.3,'load_scale':1.2},
}


def reserve_audit(env,previous_row,baseline):
    """独立检查申报模型端点及实际下一时刻的线性/AC 可行性，不把端点检查当持续履约证明。"""
    core=env.core;rows=[]
    for mode in ('min_grid','max_grid'):
        plans,meta=core.plan(objective=mode,fixed_row=previous_row)
        p=plans[0];a=p['action'];forecast=core.network.audit(previous_row,a);actual=core.network.audit(core.row(),a)
        rows.append(dict(direction=mode,optimal=bool(meta['solver_optimal']),grid_kw=grid_power(previous_row,a),
            declared_linear_violations=core.check(previous_row,a,p['ev_kw']),
            actual_linear_violations=core.check(core.row(),a,p['ev_kw']),
            declared_ac_violations=forecast['ac_violations'],actual_ac_violations=actual['ac_violations'],
            declared_ac_converged=forecast['ac_converged'],actual_ac_converged=actual['ac_converged']))
    lo,hi=[r['grid_kw'] for r in rows]
    return dict(baseline_kw=baseline,symmetric_kw=max(0,min(baseline-lo,hi-baseline)),endpoints=rows)


def run(config,dt_folder,output,seeds=tuple(range(1600,1608))):
    path=Path(output)
    if path.exists():raise ValueError('拒绝覆盖机制实验')
    path.parent.mkdir(parents=True,exist_ok=True)
    models={m:json.loads((Path(dt_folder)/(m+'.json')).read_text()) for m in ('hold','physics','residual')}
    partial=path.with_suffix(".partial.json")
    rows=json.loads(partial.read_text()) if partial.exists() else []
    completed={(r["condition"],r["method"],r["seed"]) for r in rows}
    for condition,changes in SCENARIOS.items():
        for method,model in models.items():
            for seed in seeds:
                if (condition,method,seed) in completed:continue
                c=copy.copy(config);c._cyber_record={**CyberSpec.load(c.cyber_spec).__dict__,**{k:v for k,v in changes.items() if not k.endswith('_scale')}}
                env=ResearchEnv(c,dt_model=model,vector_metrics=False,ac_safe=True)
                record=dict(condition=condition,method=method,seed=seed,failed=False,steps=[])
                try:
                    obs,_=env.reset(seed)
                    for k in ('load','wind'):
                        env.core.profiles[k+'_kw'][2:]*=changes.get(k+'_scale',1.)
                    mpc=ObservedMPC(c,env.spec,env.flex,2)
                    for t in range(c.horizon):
                        before=env.core.row().copy();truth=env.features(env.sensor_payloads())[TARGETS]
                        error=obs[0,TARGETS]-truth;start=time.perf_counter();a,meta=mpc.propose(obs[0,:54]);decision=time.perf_counter()-start
                        obs,_,_,done,i=env.step_candidate(a,np.ones(6),np.ones(6))
                        item=dict(t=t,requested_action=a.tolist(),executed_action=[i[k] for k in ('ess_power_kw','ev_charge_kw','dr_shift_kw','dr_shed_kw','pv_curtail_kw','wind_curtail_kw')],cost=i['cost']+i['terminal_penalty'],mse=float(np.mean(error**2)),
                            covered=bool(np.all(abs(error)<=np.array(model['halfwidth'])+1e-6)),aoi=i['aoi_mean_seconds'],
                            correction_kw=i['shield_l1_kw'],mpc_failed=bool(meta['mpc_failed']),decision_seconds=decision,
                            ac_violations=i['ac_violations'],ev_unmet_kwh=i['ev_unmet_kwh'],dr_backlog_kwh=i['dr_backlog_kwh'])
                        if not done:
                            try:item['reserve_audit']=reserve_audit(env,before,i['grid_power_kw'])
                            except (RuntimeError,ValueError) as exc:item['reserve_error']=str(exc)
                        record['steps'].append(item)
                    record.update(cost=sum(x['cost'] for x in record['steps']),rmse=float(np.sqrt(np.mean([x['mse'] for x in record['steps']]))))
                except (RuntimeError,ValueError) as exc:record.update(failed=True,error=str(exc))
                rows.append(record)
            print(condition,method,flush=True)
            path.with_suffix('.partial.json').write_text(json.dumps(rows,ensure_ascii=False))
    summary={}
    for condition in SCENARIOS:
        group=[r for r in rows if r['condition']==condition];pairs=[]
        for seed in seeds:
            a=next(r for r in group if r['method']=='physics' and r['seed']==seed)
            b=next(r for r in group if r['method']=='residual' and r['seed']==seed)
            if not a['failed'] and not b['failed']:pairs.append(b['cost']-a['cost'])
        summary[condition]=dict(failed_episodes=sum(r['failed'] for r in group),paired_count=len(pairs),
            residual_minus_physics_cost=stats(pairs) if pairs else None)
    result=dict(protocol='preregistered-mechanism-grid-v1',seeds=list(seeds),conditions=SCENARIOS,summary=summary,rows=rows,
        limits=['多个扰动条件的置信区间是探索性，未经多重比较校正','备用仅核验端点和下一时刻；尚非整段激活履约证明','失败回合保留，配对统计同时报告有效样本数'])
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/research_smoke.json');p.add_argument('--dt-folder',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Config.load(a.config),a.dt_folder,a.output)
