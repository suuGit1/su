"""在同一测试场景、资源快照与物理执行器上配对比较三种控制器。"""
import argparse
import json
from pathlib import Path
import torch
from .config import Config
from .runner import evaluate, write_csv
from .baselines import run


def compare(checkpoint, output, csv_path=None, episodes=3, seed=100000, lookahead=6, ev_sessions_path=None):
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('比较输出目录非空')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=Config(**state['config']); cfg._dispatch_record=state.get('dispatch_spec')
    cfg._flex_record=state.get('flex_spec'); cfg._ev_bundle=state.get('ev_bundle')
    if ev_sessions_path:
        from .flex_resources import read_bundle
        cfg._ev_bundle=read_bundle(ev_sessions_path)
    if cfg.network_model!='ieee33' or not cfg.safety:
        raise ValueError('统一可行域比较要求 IEEE33 且 safety=true；不安全消融请单独评估')
    # 学习策略先执行数据隔离检查；失败时不继续跑对照。
    learned=evaluate(checkpoint,out/'learned',episodes,seed,'cpu',csv_path,ev_sessions_path=ev_sessions_path)
    oracle=run(cfg,out/'milp_oracle','milp_oracle',episodes,seed,csv_path,lookahead)
    mpc=run(cfg,out/'mpc','mpc',episodes,seed,csv_path,lookahead)
    paired=[]
    for a,b,c in zip(learned,oracle,mpc):
        paired.append(dict(episode=a['episode'],scenario_seed=a['scenario_seed'],
            learned_objective=a['objective'],oracle_objective=b['objective'],mpc_objective=c['objective'],
            learned_minus_oracle=a['objective']-b['objective'],mpc_minus_oracle=c['objective']-b['objective'],
            learned_ac_violations=a['ac_violations'],oracle_ac_violations=b['ac_violations'],mpc_ac_violations=c['ac_violations']))
    write_csv(out/'paired.csv',paired)
    (out/'comparison.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),episodes=episodes,seed=seed,
        csv=csv_path,lookahead=lookahead,objective='无损线性电费+退化+DR+日末SOC短缺罚项',
        caveat='MILP 使用完美未来；MPC 使用持久性预测；AC 违规与 AC 成本单列，线性可行不是 AC 安全保证'),ensure_ascii=False,indent=2),encoding='utf-8')
    return paired


def main():
    p=argparse.ArgumentParser(description='MAPPO / 完美预知 MILP / 因果 MPC 配对测试')
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--csv')
    p.add_argument('--ev-sessions')
    p.add_argument('--episodes',type=int,default=3); p.add_argument('--seed',type=int,default=100000); p.add_argument('--lookahead',type=int,default=6)
    a=p.parse_args(); compare(a.checkpoint,a.output,a.csv,a.episodes,a.seed,a.lookahead,a.ev_sessions)


if __name__=='__main__': main()
