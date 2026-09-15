"""同一环境执行日前完美预知 MILP 与因果持久性预测 MPC。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import Config
from .environment import VPPAdapter
from .optimization import solve_dispatch
from .runner import write_csv


def run(config, output, controller='mpc', episodes=3, seed=100000, csv_path=None, lookahead=6):
    config.validate()
    if config.cyber_mode!='off': raise ValueError('此入口为能源 MILP/MPC；C3 固定规则公平对照请使用 vpp_mappo.cyber_compare')
    if config.network_model!='ieee33': raise ValueError('统一优化对照要求 ieee33 环境')
    if episodes<1 or seed<0 or lookahead<1 or controller not in ('mpc','milp_oracle'): raise ValueError('对照参数不合法')
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise ValueError('输出目录非空')
    env=VPPAdapter(config,csv_path)
    if env.data and episodes>len(env.data.profiles): raise ValueError('评估场景数不得超过 CSV 场景数')
    out.mkdir(parents=True,exist_ok=True)
    metadata=dict(config=asdict(config),dispatch_spec=env.spec.record(),controller=controller,lookahead=lookahead,
        data_sha256=env.data.sha256 if env.data else None,data_source='csv' if env.data else 'synthetic',
        forecast='完整实际曲线：仅作线性模型完美预知参考' if controller=='milp_oracle' else '当前测量值持久性预测：不读取未来实际值')
    if config.resource_model=='sessions_v1':
        metadata.update(resource_contract='sessions_v1',flex_spec=asdict(env.flex),ev_bundle=env.bundle,
            ev_information='全部会话' if controller=='milp_oracle' else '仅已接入会话；持久性预测延伸至日末的可行性尾部')
    metadata['optimization_objective'] = 'economic; extra objectives are evaluation only'
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    trajectories=[]; totals=[]
    for ep in range(episodes):
        env.reset(seed+ep,ep); rows=[]
        if controller=='milp_oracle':
            if config.resource_model=='sessions_v1':
                plan,meta=env.plan(oracle=True)
            else:
                plan,meta=solve_dispatch(env.spec,env.network,[env.row(t) for t in range(config.horizon)],env.soc,
                config.dt_hours,config.terminal_soc_penalty,config.solver_time_limit)
        for t in range(config.horizon):
            if controller=='mpc':
                # 每步重算，仅访问当前值。窗口末端 SOC 目标为显式启发式，不冒充完美预测。
                predicted=[env.row().copy() for _ in range(min(lookahead,config.horizon-t))]
                if config.resource_model=='sessions_v1':
                    plan,meta=env.plan(lookahead=lookahead)
                else:
                    plan,meta=solve_dispatch(env.spec,env.network,predicted,env.soc,config.dt_hours,
                    config.terminal_soc_penalty,config.solver_time_limit)
            action=plan[t] if controller=='milp_oracle' else plan[0]
            _,_,_,_,info=env.step_physical(action)
            row=dict(episode=ep+1,scenario_seed=seed+ep,step=t,**info,**meta)
            rows.append(row); trajectories.append(row)
        total=dict(episode=ep+1,scenario_seed=seed+ep,cost=sum(r['cost'] for r in rows),
            objective=sum(r['cost']+r['terminal_penalty'] for r in rows),reward=sum(r['reward'] for r in rows),
            violations=sum(r['constraint_violations'] for r in rows),ac_violations=sum(r['ac_violations'] for r in rows),
            ac_failed_steps=sum(not r['ac_converged'] for r in rows),
            ac_cost=sum(r['ac_cost'] for r in rows) if all(r['ac_converged'] for r in rows) else None,
            nonoptimal_solves=sum(not r['solver_optimal'] for r in rows) if controller=='mpc' else int(not meta['solver_optimal']))
        if config.resource_model=='sessions_v1':
            from .flex_environment import resource_totals
            total.update(resource_totals(rows))
        totals.append(total)
        if config.metrics_enabled:
            from .objectives import aggregate_metrics
            total.update(aggregate_metrics(rows))
        print(f"{controller} episode={ep+1} cost={total['cost']:.3f} AC违规={total['ac_violations']}",flush=True)
    write_csv(out/'trajectory.csv',trajectories); write_csv(out/'episodes.csv',totals)
    summary=dict(controller=controller,episodes=episodes,mean_cost=float(np.mean([r['cost'] for r in totals])),
        mean_objective=float(np.mean([r['objective'] for r in totals])),
        total_violations=sum(r['violations'] for r in totals),total_ac_violations=sum(r['ac_violations'] for r in totals),
        ac_failed_steps=sum(r['ac_failed_steps'] for r in totals))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if config.metrics_enabled:
        from .objectives import OBJECTIVE_NAMES, CONTRACT_VERSION
        if config.resource_model=='sessions_v1': CONTRACT_VERSION += '+sessions-dr-curtail-v1'
        summary['objective_contract'] = CONTRACT_VERSION
        summary['mean_objective_vector'] = [float(np.mean([r[k] for r in totals])) for k in OBJECTIVE_NAMES]
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return totals


def main():
    p=argparse.ArgumentParser(description='IEEE33 统一 MILP/MPC 对照')
    p.add_argument('--config',default='configs/ieee33_hourly.json'); p.add_argument('--output',required=True)
    p.add_argument('--controller',choices=['milp_oracle','mpc'],default='mpc'); p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--seed',type=int,default=100000); p.add_argument('--csv'); p.add_argument('--lookahead',type=int,default=6)
    a=p.parse_args(); run(Config.load(a.config),a.output,a.controller,a.episodes,a.seed,a.csv,a.lookahead)


if __name__=='__main__': main()
