"""B1：冻结策略和相同起始状态，分离偏好响应与安全执行修正。"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .train import load
from .environment import ResearchEnv


def diagnose(checkpoint,csv_path,output,days=1,states=3,preferences=None,probe_steps=2):
    import torch
    from vpp_mappo.algorithms import to_numpy
    state,c,agent=load(checkpoint)
    if state['method'] not in ('ordinary','pareto'):raise ValueError('B1只诊断独立双MAPPO')
    weights=np.asarray(preferences if preferences is not None else [[1,0,0],[0,1,0],[0,0,1],[1/3]*3],float)
    if weights.ndim!=2 or weights.shape[1]!=3 or len(weights)<2 or not np.isfinite(weights).all() or np.any(weights<0) or not np.allclose(weights.sum(1),1):raise ValueError('诊断偏好必须为至少两个三维单纯形向量')
    if not 1<=probe_steps<=c.horizon or not 1<=states<=c.horizon-probe_steps+1 or days<1:raise ValueError('日期、诊断状态数或反事实窗口非法')
    env=ResearchEnv(c,csv_path,state['dt_model'],state['robust'],ac_safe=True,resource_mode=state.get('resource_mode','joint'),reserve_mode=state['reserve_mode'])
    if not env.data or days>len(env.data.profiles):raise ValueError('必须提供足够的独立真实日期')
    reserved=set(state['train_scenarios'])
    if state['dt_model']:reserved|=set(state['dt_model']['training_scenarios'])|set(state['dt_model']['calibration_scenarios'])
    if set(env.data.scenario_names)&reserved:raise ValueError('诊断日期与策略或DT拟合/校准重叠')
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('诊断输出目录非空')
    out.mkdir(parents=True,exist_ok=True);rows=[]
    def act(obs,w,rnn):
        if state['method']=='pareto':
            a,_,v=agent.act(obs,w,True);return a,v,rnn
        with torch.no_grad():a,n=agent.policy.act(obs,rnn,np.ones((env.num_agents,1),np.float32),deterministic=True)
        return to_numpy(a),None,to_numpy(n)
    selected=set(map(int,np.linspace(0,c.horizon-probe_steps,states,dtype=int)))
    for day in range(days):
        obs,_=env.reset(60000+day,day);rnn=np.zeros((env.num_agents,1,c.hidden_size),np.float32)
        for step in range(c.horizon):
            if step in selected:
                for w in weights:
                    a,v,_=act(obs,w,rnn.copy());row=dict(date=env.data.scenario_names[day],step=step,preference=w.tolist(),
                        raw_action=a.tolist(),critic_vector=None if v is None else v.tolist(),tanh_saturation_fraction=float(np.mean(np.abs(np.tanh(a))>.95)),failed=False)
                    branch=copy.deepcopy(env);branch.preference=w
                    try:
                        next_obs,_,_,_,info=branch.step(a)
                        candidate=np.array(info['issued_energy_candidate_kw'])
                        received=np.array(info['received_energy_candidate_kw'])
                        executed=np.array([info[k] for k in ('ess_power_kw','ev_charge_kw','dr_shift_kw','dr_shed_kw','pv_curtail_kw','wind_curtail_kw')])
                        row.update(candidate_kw=candidate.tolist(),received_kw=received.tolist(),executed_kw=executed.tolist(),guarded_kw=info['interval_guard_candidate_kw'],
                            interval_correction_l1_kw=float(np.abs(np.asarray(info['interval_guard_candidate_kw'])-received).sum()),
                            local_fallback=info.get('local_fallback',False),command_latency_seconds=info.get('command_latency_seconds'),
                            correction_l1_kw=float(np.abs(executed-candidate).sum()),objective_vector=info['objective_vector'],
                            guard_certified=info['guard_certificate_survived'],violations=info['constraint_violations'],ac_violations=info['ac_violations'])
                        window=[]
                        branch_rnn=act(obs,w,rnn.copy())[2]
                        for offset in range(probe_steps):
                            if offset:
                                follow,_,branch_rnn=act(next_obs,w,branch_rnn)
                                next_obs,_,_,_,info=branch.step(follow)
                            window.append(dict(offset=offset,issued=info['issued_energy_candidate_kw'],received=info['received_energy_candidate_kw'],
                                guarded=info['interval_guard_candidate_kw'],executed=[info[k] for k in ('ess_power_kw','ev_charge_kw','dr_shift_kw','dr_shed_kw','pv_curtail_kw','wind_curtail_kw')],
                                vector=info['objective_vector'],guard_certified=info['guard_certificate_survived']))
                        row['window']=window
                    except (ValueError,RuntimeError) as exc:row.update(failed=True,error=str(exc))
                    rows.append(row)
            baseline=np.ones(3)/3;env.preference=baseline;a,_,rnn=act(obs,baseline,rnn);obs,*_=env.step(a)
    groups=[]
    for day in range(days):
        for step in sorted(selected):
            group=[r for r in rows if r['date']==env.data.scenario_names[day] and r['step']==step]
            valid=[r for r in group if not r['failed']]
            g=dict(date=env.data.scenario_names[day],step=step,complete=len(valid)==len(weights),failed=len(group)-len(valid))
            if g['complete']:
                g.update(raw_preference_spread=float(np.linalg.norm(np.ptp([r['raw_action'] for r in valid],axis=0))),
                         candidate_spread_kw=float(np.linalg.norm(np.ptp([r['candidate_kw'] for r in valid],axis=0))),
                         received_spread_kw=float(np.linalg.norm(np.ptp([r['received_kw'] for r in valid],axis=0))),
                         guarded_spread_kw=float(np.linalg.norm(np.ptp([r['guarded_kw'] for r in valid],axis=0))),
                         executed_spread_kw=float(np.linalg.norm(np.ptp([r['executed_kw'] for r in valid],axis=0))),
                         mean_correction_l1_kw=float(np.mean([r['correction_l1_kw'] for r in valid])))
                g['window_spread']=[dict(offset=offset,**{key:float(np.linalg.norm(np.ptp([r['window'][offset][key] for r in valid],axis=0)))
                    for key in ('issued','received','guarded','executed')}) for offset in range(probe_steps)]
            groups.append(g)
    result=dict(method=state['method'],checkpoint=str(checkpoint),preferences=weights.tolist(),probe_steps=probe_steps,rows=rows,groups=groups,
        scope='相同初始观测、物理状态和随机状态的多步反事实；窗口后续状态可分化；主轨迹由均匀偏好推进；普通MAPPO无显式偏好输入，执行器仍可能响应偏好；不是完整回报因果证明')
    (out/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid=[g for g in groups if g['complete']]
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for key,label in [('candidate_spread_kw','Issued'),('received_spread_kw','Received'),('guarded_spread_kw','Interval guard'),('executed_spread_kw','Executed')]:ax.plot(range(len(valid)),[g[key] for g in valid],marker='o',label=label)
    ax.set_xlabel('Matched state');ax.set_ylabel('Preference spread (kW, L2)');ax.legend()
    fig.savefig(out/'preference_response.png',dpi=160);fig.savefig(out/'preference_response.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for key in ('issued','received','guarded','executed'):
        ax.plot(range(probe_steps),[np.mean([g['window_spread'][t][key] for g in valid]) if valid else np.nan for t in range(probe_steps)],marker='o',label=key)
    ax.set_xticks(range(probe_steps));ax.set_xlabel('Counterfactual dispatch offset');ax.set_ylabel('Mean preference spread (kW, L2)');ax.legend()
    fig.savefig(out/'delayed_response.png',dpi=160);fig.savefig(out/'delayed_response.svg');plt.close(fig)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--csv',required=True)
    p.add_argument('--output',required=True);p.add_argument('--days',type=int,default=1);p.add_argument('--states',type=int,default=3)
    p.add_argument('--probe-steps',type=int,default=2)
    a=p.parse_args();diagnose(a.checkpoint,a.csv,a.output,a.days,a.states,probe_steps=a.probe_steps)
