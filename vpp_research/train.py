"""统一环境交互预算的官方普通/固定权重 MAPPO 与偏好向量 MAPPO。"""
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import torch
from vpp_mappo.algorithms import OfficialPPO,to_numpy
from vpp_mappo.runner import seed_all,write_csv
from .environment import ResearchEnv,OBS_VERSION,OBJECTIVE_VERSION,AC_OBJECTIVE_VERSION,PCC_OBJECTIVE_VERSION,SCALES
from .central_agent import CentralPPO
from .pareto_agent import ParetoAgent,sample_preference,VERSION


def train(config,dt_model,output,method='pareto',preference=(1.,0.,0.),robust=False,scenario_offset=0,resource_mode='joint',reserve_mode='linear'):
    config.validate()
    if config.algorithm!='mappo' or config.device!='cpu':raise ValueError('本研究驱动当前验证了 CPU MAPPO；旧 GPU/IPPO 入口继续独立保留')
    if method not in ('ordinary','fixed','pareto','central'):raise ValueError('未知学习方法')
    pref=np.asarray(preference,dtype=float)
    if pref.shape!=(3,) or not np.isfinite(pref).all() or np.any(pref<0) or not np.isclose(pref.sum(),1):raise ValueError('训练偏好必须在三维单纯形上')
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('训练目录非空')
    out.mkdir(parents=True,exist_ok=True);seed_all(config.seed,config.threads)
    started=time.perf_counter()
    env=ResearchEnv(config,config.train_csv,dt_model,robust,resource_mode=resource_mode,reserve_mode=reserve_mode);rng=np.random.default_rng(config.seed+31415)
    agent=(CentralPPO if method=='central' else ParetoAgent)(65,env.num_agents,config.hidden_size,config.lr) if method in ('pareto','central') else OfficialPPO(config,env)
    history=[];weights=[]
    for ep in range(config.episodes):
        w=sample_preference(rng,ep) if method=='pareto' else np.asarray(preference)
        env.preference=w;env.reward_mode='economic' if method in ('ordinary','central') else 'weighted'
        obs,share=env.reset(config.seed*10000+2000+scenario_offset+ep,ep);weights.append(w.tolist());vectors=[];violations=0
        if method in ('pareto','central'):roll=dict(obs=[],preferences=[],actions=[],logprobs=[],values=[],vectors=[],dones=[])
        else:
            buffer=agent.buffer();buffer.obs[0,0]=obs;buffer.share_obs[0,0]=share
        for t in range(config.horizon):
            if method in ('pareto','central'):a,lp,v=agent.act(obs,w)
            else:
                agent.trainer.prep_rollout()
                with torch.no_grad():
                    v,a,lp,ra,rc=agent.policy.get_actions(buffer.share_obs[t,0],buffer.obs[t,0],buffer.rnn_states[t,0],buffer.rnn_states_critic[t,0],buffer.masks[t,0])
                a=to_numpy(a)
            try:no,ns,reward,done,info=env.step(a)
            except (RuntimeError,ValueError) as exc:
                failure=dict(completed_env_steps=ep*config.horizon+env.core.t,episode=ep,step=t,scenario_seed=config.seed*10000+2000+scenario_offset+ep,
                    error=str(exc),soc=env.core.soc.tolist(),backlog=env.core.backlog,shifted=env.core.shifted,
                    shed_used=env.core.shed_used,remaining=env.core.remaining,row=env.core.row())
                (out/'failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2))
                raise
            vector=np.array(info['objective_vector']);vectors.append(vector);violations+=info['constraint_violations']
            if method in ('pareto','central'):
                # 将相同安全罚项施加于每个目标，任何和为 1 的偏好都获得同样惩罚。
                if method=='central':vector=np.repeat(vector[0],3)
                penalized=vector-config.violation_penalty*info['constraint_violations']/config.reward_scale
                for k,item in zip(roll,(obs.copy(),w.copy(),a.copy(),lp.copy(),v.copy(),penalized,done)):roll[k].append(item)
            else:
                masks=np.full((1,env.num_agents,1),0. if done else 1.,np.float32)
                buffer.insert(ns[None],no[None],to_numpy(ra)[None]*masks[...,None],to_numpy(rc)[None]*masks[...,None],
                    a[None],to_numpy(lp)[None],to_numpy(v)[None],reward[None],masks)
            obs,share=no,ns
        if method in ('pareto','central'):metrics=agent.update(roll,config.ppo_epoch,config.clip_param,config.gamma,config.gae_lambda,config.entropy_coef)
        else:
            buffer.compute_returns(np.zeros((1,env.num_agents,1),np.float32),agent.trainer.value_normalizer)
            agent.trainer.prep_training();metrics=agent.trainer.train(buffer)
            metrics={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items()}
        history.append(dict(episode=ep+1,env_steps=(ep+1)*config.horizon,preference=json.dumps(w.tolist()),
            vector=json.dumps(np.sum(vectors,axis=0).tolist()),violations=violations,**metrics))
    state=dict(version=VERSION,observation_version=OBS_VERSION,objective_version=env.objective_version,reserve_mode=reserve_mode,scales=SCALES.tolist(),
        method=method,config=asdict(config),dt_model=dt_model,robust=robust,preference=pref.tolist(),training_preferences=weights,
        resource_mode=resource_mode,training_seconds=time.perf_counter()-started,n_agents=env.num_agents,dispatch_spec=env.spec.record(),flex_spec=asdict(env.flex),cyber_spec=asdict(env.cyber_spec),ev_bundle=env.bundle,
        train_scenarios=env.data.scenario_names if env.data else [],train_fingerprints=env.data.fingerprints if env.data else [],
        train_seeds=[config.seed*10000+2000+scenario_offset+ep for ep in range(config.episodes)])
    state['agent']=agent.state_dict() if method in ('pareto','central') else agent.state()
    torch.save(state,out/'latest.pt');write_csv(out/'training.csv',history)
    (out/'metadata.json').write_text(json.dumps({k:v for k,v in state.items() if k!='agent'},ensure_ascii=False,indent=2),encoding='utf-8')
    return state


def load(checkpoint):
    from vpp_mappo.config import Config
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    expected={'linear':OBJECTIVE_VERSION,'ac_checked':AC_OBJECTIVE_VERSION,'pcc_checked':PCC_OBJECTIVE_VERSION}.get(state.get('reserve_mode','linear'))
    if (state['version'],state['observation_version'],state['objective_version'])!=(VERSION,OBS_VERSION,expected):raise ValueError('研究检查点协议不匹配')
    c=Config(**state['config']);c._dispatch_record=state['dispatch_spec'];c._flex_record=state['flex_spec'];c._cyber_record=state['cyber_spec'];c._ev_bundle=state['ev_bundle']
    c.train_csv=None;env=ResearchEnv(c,dt_model=state['dt_model'],robust=state['robust'],resource_mode=state.get('resource_mode','joint'),vector_metrics=False,reserve_mode=state.get('reserve_mode','linear'))
    agent=(CentralPPO if state['method']=='central' else ParetoAgent)(65,env.num_agents,c.hidden_size,c.lr) if state['method'] in ('pareto','central') else OfficialPPO(c,env)
    if state['method'] in ('pareto','central'):agent.load_state_dict(state['agent']);agent.eval()
    else:agent.load(state['agent']);agent.trainer.prep_rollout()
    return state,c,agent


def evaluate(checkpoint,preferences,seeds,output=None,robust=None,stress=None,csv_path=None,ev_bundle=None,ac_safe=False):
    state,c,agent=load(checkpoint)
    if set(seeds)&set(state['train_seeds']):raise ValueError('策略训练与评估种子重叠')
    if ev_bundle is not None:c._ev_bundle=ev_bundle
    if stress:c._cyber_record={**c._cyber_record,**{k:v for k,v in stress.items() if k not in ('wind_scale','load_scale')}}
    env=ResearchEnv(c,csv_path,state['dt_model'],state['robust'] if robust is None else robust,ac_safe=ac_safe,resource_mode=state.get("resource_mode","joint"),reserve_mode=state.get("reserve_mode","linear"))
    if env.data and (set(env.data.scenario_names)&set(state['train_scenarios']) or set(env.data.fingerprints)&set(state['train_fingerprints'])):raise ValueError('真实测试数据与训练重叠')
    if env.data and len(seeds)>len(env.data.profiles):raise ValueError('真实测试场景数量不足')
    reserved=set(state['dt_model']['training_scenarios'])|set(state['dt_model']['calibration_scenarios']) if state['dt_model'] else set()
    if env.data and set(env.data.scenario_names)&reserved:raise ValueError('真实评估场景与 DT 拟合/校准场景重叠')
    if {f'synthetic:{s}' for s in seeds}&reserved:raise ValueError('评估数据与 DT 拟合/校准场景重叠')
    results=[]
    for preference in preferences:
        w=np.asarray(preference,dtype=float)
        if w.shape!=(3,) or np.any(w<0) or not np.isclose(w.sum(),1):raise ValueError('偏好必须在三维单纯形上')
        for index,seed in enumerate(seeds):
            row=dict(seed=seed,preference=w.tolist(),unseen_preference=not any(np.allclose(w,p,atol=1e-9,rtol=0) for p in state['training_preferences']),failed=False,env_steps=0)
            try:
                env.preference=w;env.reward_mode="economic" if state["method"]=="ordinary" else "weighted"
                obs,_=env.reset(seed,index)
                if stress:
                    for name in ('wind','load'):
                        if name+'_scale' in stress:env.core.profiles[name+'_kw'][2:]*=stress[name+'_scale']
                rnn=np.zeros((env.num_agents,1,c.hidden_size),np.float32);masks=np.ones((env.num_agents,1),np.float32);infos=[]
                inference_times=[]
                for _ in range(c.horizon):
                    started=time.perf_counter()
                    if state['method'] in ('pareto','central'):a,_,_=agent.act(obs,w,True)
                    else:
                        with torch.no_grad():a,r=agent.policy.act(obs,rnn,masks,deterministic=True)
                        a=to_numpy(a);rnn=to_numpy(r)
                    inference_times.append(time.perf_counter()-started)
                    obs,_,_,_,info=env.step(a);infos.append(info);row['env_steps']+=1
                row.update(objective_version=state['objective_version'],emergency_steps=sum(i.get('emergency',False) for i in infos),
                    emergency_service_degraded_steps=sum(i.get('emergency_service_degraded',False) for i in infos),inference_seconds=inference_times,training_seconds=state.get('training_seconds'),vector=np.sum([i['objective_vector'] for i in infos],axis=0).tolist(),
                    violations=sum(i['constraint_violations'] for i in infos),ac_violations=sum(i['ac_violations'] for i in infos),
                    ac_failed=sum(not i['ac_converged'] for i in infos),reserve_invalid=sum(not i['reserve_valid'] for i in infos),
                    ev_unmet_kwh=sum(i['ev_unmet_kwh'] for i in infos),dr_backlog_kwh=infos[-1]['dr_backlog_kwh'],
                    ac_safety_seconds=sum(i.get('ac_safety_seconds',0) for i in infos),
                    ac_safety_attempts=sum(i.get('ac_safety_attempts',0) for i in infos),
                    interval_covered_steps=sum(i['interval_covered'] for i in infos),
                    estimation_rmse=float(np.sqrt(np.mean([i['research_dt_mse'] for i in infos]))),
                    safety_interventions=sum(i['shield_l1_kw']>1e-5 for i in infos),guard_infeasible=sum(not i['guard_feasible'] for i in infos) if env.robust else None,
                    guard_certified_steps=sum(i['guard_certificate_survived'] for i in infos),mean_aoi=float(np.mean([i['aoi_mean_seconds'] for i in infos])))
            except (RuntimeError,ValueError) as exc:row.update(failed=True,error=str(exc),vector=None,env_steps=max(row['env_steps'],env.core.t))
            results.append(row)
    if output:Path(output).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    return results
