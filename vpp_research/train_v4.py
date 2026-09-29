"""完整多日批次更新；官方MAPPO保留独立基线，固定验证额外交互单列。"""
from dataclasses import asdict
from pathlib import Path
import json
import os
import random
import time
import numpy as np
import torch
from vpp_mappo.algorithms import OfficialPPO,to_numpy
from vpp_mappo.runner import seed_all,write_csv
from .environment import ResearchEnv,OBS_VERSION_V4,SCALES
from .pareto_agent import sample_preference
from .pareto_v4 import ParetoV4,VERSION
from .central_agent import CentralPPO
from .ledger import Ledger
from .trace import append,step_record,plain


def make_agent(config,env,method):
    if method=='pareto':return ParetoV4(env.obs_dim,env.num_agents,config.hidden_size,config.lr,config.target_kl)
    if method=='central':return CentralPPO(env.obs_dim,env.num_agents,config.hidden_size,config.lr)
    return OfficialPPO(config,env)


def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),torch=torch.get_rng_state(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]))


def restore_rng(state):
    random.setstate(state['python']);torch.set_rng_state(state['torch']);n=state['numpy']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))


def train_v4(config,dt_model,output,method='pareto',preference=(1.,0.,0.),robust=False,
             scenario_offset=0,resource_mode='joint',reserve_mode='linear',resume=False,trace_path=None,validation_csv=None):
    config.validate()
    if config.device!='cpu' or config.algorithm!='mappo':raise ValueError('v4研究入口使用CPU MAPPO')
    if method not in ('ordinary','fixed','central','pareto'):raise ValueError('未知学习方法')
    pref=np.asarray(preference,dtype=float)
    if pref.shape!=(3,) or not np.isfinite(pref).all() or min(pref)<0 or not np.isclose(pref.sum(),1):raise ValueError('偏好非法')
    out=Path(output)
    if out.exists() and any(out.iterdir()) and not resume:raise ValueError('训练目录非空')
    out.mkdir(parents=True,exist_ok=True);seed_all(config.seed,config.threads)
    env=ResearchEnv(config,config.train_csv,dt_model,robust,resource_mode=resource_mode,reserve_mode=reserve_mode)
    agent=make_agent(config,env,method);custom=method in ('pareto','central');rng=np.random.default_rng(config.seed+31415)
    contract=dict(config=asdict(config),dt_model=dt_model,method=method,preference=pref.tolist(),reserve_mode=reserve_mode,
        resource_mode=resource_mode,scenario_offset=scenario_offset,robust=env.robust,dispatch=env.spec.record(),flex=asdict(env.flex),
        cyber=asdict(env.cyber_spec),ev_bundle=env.bundle,fingerprints=env.data.fingerprints if env.data else [],validation_csv=str(validation_csv))
    if validation_csv:
        import hashlib
        contract['validation_sha256']=hashlib.sha256(Path(validation_csv).read_bytes()).hexdigest()
    first=0;history=[];weights=[];attempts=[];validation_history=[];prior_seconds=0.;validation_seconds=0.;best_score=None
    ledger=Ledger(out/'interactions.sqlite')
    if resume:
        saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        if saved['contract']!=contract:raise ValueError('v4续训协议改变，请新建目录')
        first=saved['episode'];history=saved['history'];weights=saved['weights'];attempts=saved['episode_attempts']
        validation_history=saved['validation_history'];best_score=saved['best_score'];prior_seconds=saved['training_seconds']
        ledger.reconcile(attempts,config.horizon)
        if custom:agent.load_state_dict(saved['agent'])
        else:agent.load(saved['agent'])
        if method=='pareto':agent.load_optimizer_state(saved['optimizer'])
        elif method=='central':agent.optimizer.load_state_dict(saved['optimizer'])
        else:
            agent.policy.actor_optimizer.load_state_dict(saved['agent']['actor_optimizer']);agent.policy.critic_optimizer.load_state_dict(saved['agent']['critic_optimizer'])
        rng.bit_generator.state=saved['preference_rng'];restore_rng(saved['rng'])
    attempt_id=str(time.time_ns());started=time.perf_counter()
    def state():
        return dict(version=VERSION,observation_version=OBS_VERSION_V4,objective_version=env.objective_version,reserve_mode=reserve_mode,
            scales=SCALES.tolist(),method=method,config=asdict(config),dt_model=dt_model,robust=env.robust,preference=pref.tolist(),
            training_preferences=weights,resource_mode=resource_mode,training_seconds=prior_seconds+time.perf_counter()-started-validation_seconds,
            n_agents=env.num_agents,dispatch_spec=env.spec.record(),flex_spec=asdict(env.flex),cyber_spec=asdict(env.cyber_spec),ev_bundle=env.bundle,
            train_scenarios=env.data.scenario_names if env.data else [],train_fingerprints=env.data.fingerprints if env.data else [],
            train_seeds=[config.seed*10000+2000+scenario_offset+ep for ep in range(config.episodes)],agent=agent.state_dict() if custom else agent.state())
    try:
        for begin in range(first,config.episodes,config.rollout_episodes):
            end=min(config.episodes,begin+config.rollout_episodes);length=(end-begin)*config.horizon
            roll={k:[] for k in ('obs','preferences','actions','logprobs','values','vectors','dones')}
            if not custom:agent.args.episode_length=length;buffer=agent.buffer()
            for ep in range(begin,end):
                w=sample_preference(rng,ep) if method=='pareto' else pref.copy();weights.append(w.tolist())
                env.preference=w;env.reward_mode='economic' if method in ('ordinary','central') else 'weighted'
                # 各方法共享按种子确定的训练日期洗牌，不读取测试集。
                n=len(env.data.profiles) if env.data else 1;index=ep+scenario_offset
                day=int(np.random.default_rng(config.seed+7919*(index//n+1)).permutation(n)[index%n])
                obs,share=env.reset(config.seed*10000+2000+scenario_offset+ep,day);vectors=[];infos=[]
                for t in range(config.horizon):
                    k=(ep-begin)*config.horizon+t
                    if custom:a,lp,v=agent.act(obs,w)
                    else:
                        buffer.obs[k,0]=obs;buffer.share_obs[k,0]=share;agent.trainer.prep_rollout()
                        with torch.no_grad():v,a,lp,ra,rc=agent.policy.get_actions(buffer.share_obs[k,0],buffer.obs[k,0],buffer.rnn_states[k,0],buffer.rnn_states_critic[k,0],buffer.masks[k,0])
                        a=to_numpy(a)
                    ident=ledger.begin(attempt_id,ep,t)
                    try:no,ns,reward,done,info=env.step(a)
                    except Exception as exc:ledger.finish(ident,'error',str(exc));raise
                    record=step_record(env,obs,a,w,info,attempt_id=attempt_id,episode=ep,step=t,phase='train');ledger.finish(ident,'success',record)
                    append(out/'attempts.jsonl',dict(event='step',attempt_id=attempt_id,episode=ep,step=t))
                    if trace_path:append(trace_path,record)
                    vector=np.array(info['objective_vector']);vectors.append(vector);infos.append(info)
                    if custom:
                        rv=np.repeat(vector[0],3) if method=='central' else vector
                        rv=rv-config.violation_penalty*info['constraint_violations']/config.reward_scale
                        for key,item in zip(roll,(obs.copy(),w.copy(),a.copy(),lp.copy(),v.copy(),rv,done)):roll[key].append(item)
                    else:
                        masks=np.full((1,env.num_agents,1),0. if done else 1.,np.float32)
                        buffer.insert(ns[None],no[None],to_numpy(ra)[None]*masks[...,None],to_numpy(rc)[None]*masks[...,None],a[None],to_numpy(lp)[None],to_numpy(v)[None],reward[None],masks)
                    obs,share=no,ns
                history.append(dict(episode=ep+1,env_steps=(ep+1)*config.horizon,training_day=day,preference=json.dumps(w.tolist()),
                    vector=json.dumps(np.sum(vectors,axis=0).tolist()),terminal_penalty=sum(i['terminal_penalty'] for i in infos),
                    ac_operating_cost=sum(i['ac_cost'] for i in infos),curtailment_kwh=sum(i['pv_curtail_kw']+i['wind_curtail_kw'] for i in infos)*config.dt_hours,
                    requested_executed_l1_kw=sum(abs(i['received_energy_candidate_kw'][0]-i['ess_power_kw']) for i in infos),reserve_unconfirmed=sum(not i['reserve_valid'] for i in infos)))
            if custom:metrics=agent.update(roll,config.ppo_epoch,config.clip_param,config.gamma,config.gae_lambda,config.entropy_coef)
            else:
                buffer.compute_returns(np.zeros((1,env.num_agents,1),np.float32),agent.trainer.value_normalizer);agent.trainer.prep_training()
                metrics=agent.trainer.train(buffer);metrics={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in metrics.items()}
            for row in history[-(end-begin):]:row.update(update_env_steps=end*config.horizon,**metrics)
            snapshot=state();atomic_save(snapshot,out/'latest.pt')
            due=config.validation_every and (end==config.episodes or end//config.validation_every>begin//config.validation_every)
            if due and validation_csv:
                from .train import evaluate
                from .suite import VALIDATION_WEIGHTS,feasible,points
                from vpp_mappo.pareto import hypervolume
                before=rng_state();validation_started=time.perf_counter()
                try:rows=evaluate(out/'latest.pt',VALIDATION_WEIGHTS,range(50000,50000+config.validation_days),csv_path=validation_csv,trace_dir=out/'learning_validation'/f'episode_{end}')
                finally:validation_seconds+=time.perf_counter()-validation_started;restore_rng(before)
                good=[r for r in rows if feasible(r)];hv=hypervolume(points(rows,config.validation_days),config.selection_hv_reference)
                utility=float(np.mean([np.asarray(r['vector'])@(np.array([1,0,0]) if method in ('ordinary','central') else pref) for r in good])) if good else -1e30
                score=(len(good),hv if method=='pareto' else utility)
                validation_history.append(dict(episode=end,env_steps=end*config.horizon,hv=hv,feasible_days=len(good),total_days=len(rows),selection_steps=sum(r['env_steps'] for r in rows),results=rows))
                if best_score is None or score>tuple(best_score):best_score=score;atomic_save(snapshot,out/'best_validation.pt')
            attempts.extend([attempt_id]*(end-begin));history=json.loads(json.dumps(history,default=plain))
            saved=dict(contract=contract,episode=end,history=history,weights=weights,episode_attempts=attempts,agent=snapshot['agent'],
                optimizer=agent.optimizer_state() if method=='pareto' else agent.optimizer.state_dict() if method=='central' else None,
                rng=rng_state(),preference_rng=rng.bit_generator.state,validation_history=validation_history,best_score=best_score,training_seconds=snapshot['training_seconds'])
            atomic_save(saved,out/'resume.pt')
            for ep in range(begin,end):ledger.commit_episode(ep,attempt_id,config.horizon)
            write_csv(out/'training.csv',history)
            (out/'validation_curve.json').write_text(json.dumps(validation_history,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'[v4 {method} 种子={config.seed}] 更新至 {end}/{config.episodes} 回合，{length} 个新交互',flush=True)
        final=state();atomic_save(final,out/'latest.pt')
        (out/'metadata.json').write_text(json.dumps({k:v for k,v in final.items() if k!='agent'},ensure_ascii=False,indent=2),encoding='utf-8')
        ledger.export();(out/'budget_ledger.json').write_text(json.dumps(ledger.summary(),ensure_ascii=False,indent=2),encoding='utf-8')
        return final
    finally:ledger.close()
