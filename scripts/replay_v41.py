"""重放已记录的发令与资源分配，定位安全修订影响；不作为重新训练性能证据。"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_research.train import load
from vpp_research.environment import ResearchEnv
from vpp_research.trace import append,step_record


def run(trace,checkpoint,csv_path,output,revision):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('重放目录非空')
    out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in Path(trace).read_text(encoding='utf-8').splitlines()]
    state,c,_=load(checkpoint);c.safety_revision=revision
    env=ResearchEnv(c,csv_path=csv_path,dt_model=state['dt_model'],reserve_mode='pcc_checked')
    index=env.data.scenario_names.index(rows[0]['scenario_date']);env.preference=np.asarray(rows[0]['preference'])
    obs,_=env.reset(rows[0]['scenario_seed'],index);infos=[]
    for step,old in enumerate(rows):
        before=obs.copy()
        # 资源权重由原始策略恢复，避免把协调后的分配当成原始请求再次协调。
        raw=np.asarray(old['raw_policy_action'])
        floor=np.logaddexp(0,-2.)
        bw=np.maximum(0,np.logaddexp(0,raw[6:12,0])-floor) if len(raw)==18 else np.ones(6)
        cpu=np.maximum(0,np.logaddexp(0,raw[12:18,0])-floor) if len(raw)==18 else np.ones(6)
        obs,_,_,done,info=env.step_candidate(old['issued_energy_candidate'],bw,cpu)
        infos.append(info);append(out/'trajectory.jsonl',step_record(env,before,raw,env.preference,info,step=step))
    summary=dict(revision=revision,steps=len(infos),completed=done,reserve_invalid=sum(not i['reserve_valid'] for i in infos),ac_violations=sum(i['ac_violations'] for i in infos),certified_steps=sum(i['guard_certificate_survived'] for i in infos),scope='固定原始发令重放，不是闭环策略性能评价')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trace','checkpoint','csv','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--revision',type=int,choices=[0,1],default=1);a=p.parse_args();run(a.trace,a.checkpoint,a.csv,a.output,a.revision)
