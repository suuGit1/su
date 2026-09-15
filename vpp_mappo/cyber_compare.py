"""同一观察接口、通信计算规则和本地执行器下的 MAPPO/MPC 配对评估。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .cyber_environment import CyberVPPAdapter,cyber_totals
from .flex_environment import resource_totals
from .observed_mpc import ObservedMPC
from .runner import evaluate,write_csv


def compare(checkpoint,output,csv_path=None,episodes=3,seed=100000,lookahead=6,ev_sessions_path=None):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('输出目录非空')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=Config(**state['config'])
    # 固定通信计算规则才能隔离能源控制差异；联合资源优化须另设分组实验。
    if cfg.cyber_mode!='fixed' or not cfg.safety:raise ValueError('公平能源对照要求 cyber_mode=fixed 且 safety=true；请单独训练该配置')
    if episodes<1 or seed<0 or lookahead<1:raise ValueError('比较参数不合法')
    cfg._dispatch_record=state.get('dispatch_spec');cfg._flex_record=state.get('flex_spec')
    cfg._ev_bundle=state.get('ev_bundle');cfg._cyber_record=state.get('cyber_spec')
    if ev_sessions_path:
        from .flex_resources import read_bundle
        cfg._ev_bundle=read_bundle(ev_sessions_path)

    class AuditEnv(CyberVPPAdapter):
        def __init__(self,config,csv_path=None):
            super().__init__(config,csv_path);self.planner=ObservedMPC(config,self.spec,self.flex,lookahead)
        def step(self,actions):
            # 影子 MPC 接收当前策略所见的完全相同观察，只记录，不执行。
            frame=self.encode()[0][0].copy()
            candidate,meta=self.planner.propose(frame)
            obs,share,reward,done,info=super().step(actions)
            info.update(controller_observation_json=json.dumps(frame.tolist()),shadow_mpc_action_json=json.dumps(candidate.tolist()),**{'shadow_'+k:v for k,v in meta.items()})
            return obs,share,reward,done,info

    # 复用已有检查点校验、训练/测试日期与曲线隔离；失败则不启动 MPC 对照。
    learned=evaluate(checkpoint,out/'learned',episodes,seed,'cpu',csv_path,env_factory=AuditEnv,ev_sessions_path=ev_sessions_path)
    cfg.train_csv=None;cfg.device='cpu';cfg.validate()
    env=CyberVPPAdapter(cfg,csv_path);planner=ObservedMPC(cfg,env.spec,env.flex,lookahead)
    trajectories=[];mpc=[]
    for ep in range(episodes):
        obs,_=env.reset(seed+ep,ep);rows=[]
        for t in range(cfg.horizon):
            frame=obs[0].copy()
            candidate,meta=planner.propose(frame)
            obs,_,_,_,info=env.step_candidate(candidate,np.ones(6),np.ones(6))
            rows.append(dict(episode=ep+1,scenario_seed=seed+ep,step=t,controller_observation_json=json.dumps(frame.tolist()),**info,**meta))
        trajectories.extend(rows)
        total=dict(episode=ep+1,scenario_seed=seed+ep,cost=sum(r['cost'] for r in rows),
            objective=sum(r['cost']+r['terminal_penalty'] for r in rows),
            violations=sum(r['constraint_violations'] for r in rows),ac_violations=sum(r['ac_violations'] for r in rows),
            ac_failed_steps=sum(not r['ac_converged'] for r in rows),
            mpc_failures=sum(r['mpc_failed'] for r in rows),mpc_nonoptimal_steps=sum(not r['mpc_optimal'] for r in rows),
            mpc_solver_seconds=sum(r['mpc_solver_seconds'] or 0 for r in rows),**resource_totals(rows),**cyber_totals(rows))
        mpc.append(total)
    (out/'mpc').mkdir();write_csv(out/'mpc/trajectory.csv',trajectories);write_csv(out/'mpc/episodes.csv',mpc)
    paired=[dict(episode=a['episode'],scenario_seed=a['scenario_seed'],learned_objective=a['objective'],mpc_objective=b['objective'],
        learned_minus_mpc=a['objective']-b['objective'],learned_violations=a['violations'],mpc_violations=b['violations'],
        learned_ac_violations=a['ac_violations'],mpc_ac_violations=b['ac_violations'],mpc_failures=b['mpc_failures'],
        learned_safety_interventions=a['safety_intervention_steps'],mpc_safety_interventions=b['safety_intervention_steps']) for a,b in zip(learned,mpc)]
    write_csv(out/'paired.csv',paired)
    summary=dict(checkpoint=str(checkpoint),episodes=episodes,seed=seed,lookahead=lookahead,
        observation='same_54_features_no_private_EV_records',resource_rule='same_fixed_allocations_and_coordinator',
        statewise_check='shadow_MPC_on_exact_MAPPO_observation_recorded_in_learned_trajectory',
        caveat='闭环动作不同导致后续状态/缓存不同；同等信息权限不表示同一后续轨迹。MPC 为最近期限聚合近似，不是逐车完整信息 MPC。',
        safety='same_local_current_sensor_shield',mpc_failures=sum(r['mpc_failures'] for r in mpc),paired=paired)
    (out/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--csv');p.add_argument('--ev-sessions')
    p.add_argument('--episodes',type=int,default=3);p.add_argument('--seed',type=int,default=100000);p.add_argument('--lookahead',type=int,default=6)
    a=p.parse_args();print(json.dumps(compare(a.checkpoint,a.output,a.csv,a.episodes,a.seed,a.lookahead,a.ev_sessions),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
