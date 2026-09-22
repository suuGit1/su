"""延迟命令、控制计算抢占 DT 预算与现场后备的闭环干预验证。"""
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.observed_mpc import ObservedMPC
from .environment import ResearchEnv
from .timing import TimingContract,summarize


def run(model,output):
    c=Config.load('configs/research_smoke.json');rows=[]
    cases=dict(instant=None,delayed=TimingContract(deadline_seconds=1800),expired=TimingContract(deadline_seconds=1),outage=TimingContract(downlink_bps=0,deadline_seconds=1800))
    for name,contract in cases.items():
        for seed in range(1800,1808):
            e=ResearchEnv(c,dt_model=model,ac_safe=True,timing_contract=contract,control_cycles=1.e9 if contract else 0.)
            row=dict(case=name,seed=seed,failed=False,steps=[])
            try:
                obs,_=e.reset(seed);planner=ObservedMPC(c,e.spec,e.flex,2)
                for _ in range(c.horizon):
                    start=time.perf_counter();a,_=planner.propose(obs[0,:54]);host=time.perf_counter()-start
                    obs,_,_,_,i=e.step_candidate(a,np.ones(6),np.ones(6))
                    row['steps'].append(dict(cost=i['cost']+i['terminal_penalty'],host_decision_seconds=host,host_safety_seconds=i['ac_safety_seconds'],
                        simulated_command_latency=i.get('command_latency_seconds'),fallback=i.get('local_fallback',False),
                        missed=i.get('command_deadline_missed',False),expired=i.get('command_expired',0),ac_violations=i['ac_violations'],
                        aoi=i['aoi_mean_seconds'],cyber_energy_j=i['cyber_energy_j']))
            except (RuntimeError,ValueError) as exc:row.update(failed=True,error=str(exc))
            rows.append(row)
    summary={name:dict(episodes=len([r for r in rows if r['case']==name]),failures=sum(r['failed'] for r in rows if r['case']==name),
        host_decision=summarize([x['host_decision_seconds'] for r in rows if r['case']==name for x in r['steps']]),
        fallback_steps=sum(x['fallback'] for r in rows if r['case']==name for x in r['steps'])) for name in cases}
    Path(output).write_text(json.dumps(dict(summary=summary,rows=rows,limits=['CPU 周期、下行速率、传播时间为假设；墙钟单列','动作只在调度边界生效，不能用 15 分钟模型证明毫秒级控制','后备沿用原服务约束，仍无解时显式停止']),ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(json.loads(Path(a.model).read_text()),a.output)
