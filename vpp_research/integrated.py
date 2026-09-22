"""IEEE33、协调器、C3、DT、安全执行器与双 MAPPO 的集成运行入口。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from vpp_mappo.config import Config
from vpp_mappo.observed_mpc import ObservedMPC
from vpp_mappo.algorithms import to_numpy
from .dt import fit,assess,save
from .validate_dt import collect
from .environment import ResearchEnv
from .train import train,load
from .trace import append,step_record


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def fresh(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise ValueError('输出目录非空：'+str(path))
    path.mkdir(parents=True,exist_ok=True)
    return path


def prepare(config,out):
    out=fresh(out)
    if config.train_csv:raise ValueError('演示 DT 准备使用合成独立场景；真实数据请使用独立训练/校准/测试数据流程')
    splits={'train':list(range(1000,1006)),'calibration':list(range(1100,1112)),'test':list(range(1200,1203))}
    data={k:collect(config,v,'physics') for k,v in splits.items()}
    results={}
    for method in ('physics','residual'):
        model=fit(data['train'],data['calibration'],method=method)
        save(model,out/(method+'.json'));results[method]=assess(model,data['test'])
    write(out/'datasets.json',data)
    write(out/'assessment.json',dict(splits=splits,results=results,scope='合成功能示例；分布漂移与策略改变后覆盖率须重新验证'))
    return model


def rollout(config,model,out,method='mpc',checkpoint=None,preference=(.2,.3,.5),seed=9103,fault=None):
    out=fresh(out);w=np.asarray(preference,dtype=float)
    if w.shape!=(3,) or not np.isfinite(w).all() or min(w)<0 or not np.isclose(w.sum(),1):raise ValueError('偏好必须是三维概率向量')
    if method!='mpc':
        state,config,agent=load(checkpoint)
        if state['method']!=method:raise ValueError('模型算法与请求模式不一致')
        if seed in state['train_seeds']:raise ValueError('闭环测试种子与训练重叠')
        model=state['dt_model']
        if config.network_model!='ieee33' or config.cyber_mode!='joint' or config.coordinator_mode!='schedule' or state['reserve_mode']!='pcc_checked':
            raise ValueError('该模型不满足集成 IEEE33/C3/协调器/PCC 协议')
        if model is None:raise ValueError('集成模型必须包含校准 DT')
    if model and f'synthetic:{seed}' in model['training_scenarios']+model['calibration_scenarios']:
        raise ValueError('闭环测试与 DT 拟合/校准重叠')
    if fault=='packet_loss':
        from vpp_mappo.cyber import CyberSpec
        from dataclasses import asdict
        config._cyber_record={**asdict(CyberSpec.load(getattr(config,'_cyber_record',None) or config.cyber_spec)),'packet_loss':1.}
    if fault=='early_departure':
        key=f'synthetic:{seed}'
        config._ev_bundle=dict(schema_version=1,source='合成提前离站验收',energy_basis='grid_kwh',
            horizon=config.horizon,dt_hours=config.dt_hours,
            scenarios={key:[dict(id='early_demo',arrival_step=0,departure_step=config.horizon,
                energy_kwh=min(12.,7.*config.dt_hours*config.horizon),max_kw=11.)]},
            actual_departure_events={key:[dict(id='early_demo',departure_step=1)]})
    env=ResearchEnv(config,dt_model=model,reserve_mode='pcc_checked')
    env.preference=w;env.reward_mode='economic' if method=='ordinary' else 'weighted'
    obs,_=env.reset(seed);mpc=ObservedMPC(config,env.spec,env.flex,2)
    rnn=np.zeros((env.num_agents,1,config.hidden_size),np.float32);masks=np.ones((env.num_agents,1),np.float32)
    summary=dict(method=method,scenario_seed=seed,fault=fault,completed=False,steps=0,
        cost=0.,carbon_kg=0.,reserve_kwh=0.,ac_violations=0,constraint_violations=0,service_violations=0,packet_drops=0,
        emergency_steps=0,dt_updates=0,ev_unmet_kwh=0.,early_departures=0,terminal_dr_kwh=0.,checkpoint=str(checkpoint) if checkpoint else None)
    write(out/'config.json',config.__dict__)
    try:
        for step in range(config.horizon):
            before=obs.copy();raw=None;planner={}
            if method=='mpc':energy,planner=mpc.propose(obs[0,:54])
            elif method=='pareto':raw,_,_=agent.act(obs,w,True)
            else:
                with torch.no_grad():raw,new_rnn=agent.policy.act(obs,rnn,masks,deterministic=True)
                raw=to_numpy(raw);rnn=to_numpy(new_rnn)
            # 故障只注入现场求解器；MPC 候选仍使用其原有公开观察。
            from contextlib import nullcontext
            from unittest.mock import patch
            from vpp_mappo.optimization import DispatchInfeasible
            context=patch.object(env.core,'plan',side_effect=DispatchInfeasible('集成验收：求解超时')) if fault=='solver_timeout' and step==0 else nullcontext()
            with context:
                obs,_,_,done,info=env.step_candidate(energy,np.ones(6),np.ones(6)) if method=='mpc' else env.step(raw)
            append(out/'trajectory.jsonl',step_record(env,before,raw,w,info,phase='rollout',step=step,planner=planner))
            summary['steps']+=1;summary['cost']+=info['objective_cost']+info['terminal_penalty']
            summary['carbon_kg']+=info['carbon_kg'];summary['reserve_kwh']+=info['flexibility_kwh']
            summary['ac_violations']+=info['ac_violations'];summary['constraint_violations']+=info['constraint_violations']
            summary['service_violations']+=int(info['ev_unmet_kwh']>1e-6 or (done and abs(info['dr_backlog_kwh'])>1e-6))
            summary['packet_drops']+=info['packet_drops'];summary['emergency_steps']+=int(info.get('emergency',False))
            summary['dt_updates']+=info['dt_updates']
            summary['ev_unmet_kwh']+=info['ev_unmet_kwh'];summary['early_departures']+=info.get('early_ev_departures',0)
            summary['terminal_dr_kwh']=info['dr_backlog_kwh']
            write(out/'summary.json',summary)
        summary['completed']=bool(done)
    except Exception as exc:
        summary['error']=str(exc);write(out/'summary.json',summary);raise
    write(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','run','all'])
    p.add_argument('--config',default='configs/integrated_ieee33.json')
    p.add_argument('--output',required=True)
    p.add_argument('--dt-model');p.add_argument('--checkpoint')
    p.add_argument('--method',choices=['mpc','ordinary','pareto'],default='mpc')
    p.add_argument('--episodes',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,default=9103)
    p.add_argument('--preference',type=float,nargs=3,default=[.2,.3,.5])
    p.add_argument('--fault',choices=['packet_loss','solver_timeout','early_departure'])
    a=p.parse_args();c=Config.load(a.config)
    if a.episodes is not None:c=replace(c,episodes=a.episodes).validate()
    if c.network_model!='ieee33' or c.cyber_mode!='joint' or c.coordinator_mode!='schedule' or not c.safety:
        p.error('集成入口要求 IEEE33、joint C3、schedule 协调器和 safety=true')
    out=Path(a.output)
    if a.command=='prepare':prepare(c,out)
    elif a.command=='all':
        fresh(out);model=prepare(c,out/'dt');rows=[]
        rows.append(rollout(c,model,out/'mpc',seed=a.seed))
        for method in ('ordinary','pareto'):
            train(c,model,out/method,method=method,reserve_mode='pcc_checked',trace_path=out/method/'trajectory.jsonl')
            rows.append(rollout(c,model,out/(method+'_test'),method,out/method/'latest.pt',a.preference,a.seed))
        write(out/'summary.json',dict(status='completed',runs=rows,scope='完整架构功能运行；不是长预算优势实验'))
        from .acceptance import verify
        verify(out)
    elif a.command=='train':
        if a.method=='mpc' or not a.dt_model:p.error('训练要求 ordinary/pareto 与 --dt-model')
        model=json.loads(Path(a.dt_model).read_text())
        train(c,model,out,method=a.method,reserve_mode='pcc_checked',resume=a.resume,trace_path=out/'trajectory.jsonl')
    else:
        if a.method!='mpc' and not a.checkpoint:p.error('策略运行必须给出 --checkpoint')
        if a.method=='mpc' and not a.dt_model:p.error('MPC 闭环必须给出 --dt-model')
        model=json.loads(Path(a.dt_model).read_text()) if a.dt_model else None
        rollout(c,model,out,a.method,a.checkpoint,a.preference,a.seed,a.fault)
    print('已完成：',out.resolve())


if __name__=='__main__':main()
