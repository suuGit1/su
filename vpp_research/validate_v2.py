"""v2独立日期DT估计与相同MPC闭环对照；如无收益也保留结果。"""
import json
from pathlib import Path
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.observed_mpc import ObservedMPC
from .real_inputs import connect
from .validate_dt import collect
from .dt import fit,assess
from .environment import ResearchEnv


def run(folder,data_root='data/real',days=3):
    root=Path(folder);plan=json.loads((root/'manifest.json').read_text())
    from dataclasses import fields
    known={f.name for f in fields(Config)}
    config=Config(**{k:v for k,v in plan['config'].items() if k in known})
    for k,v in plan['config'].items():
        if k.startswith('_'):setattr(config,k,v)
    _,paths,_,_=connect(data_root,config)
    datasets=json.loads((root/'dt/datasets.json').read_text())
    models={'physics':fit(datasets['train'],datasets['validation'],method='physics'),
            'residual':json.loads((root/'dt/residual.json').read_text())}
    records=collect(config,range(61000,61000+days),'physics',paths['test'])
    results={}
    for name,model in models.items():
        rows=[]
        for day in range(days):
            e=ResearchEnv(config,paths['test'],model,vector_metrics=False,ac_safe=True)
            obs,_=e.reset(61000+day,day);mpc=ObservedMPC(config,e.spec,e.flex,2)
            infos=[]
            for _ in range(config.horizon):
                a,_=mpc.propose(obs[0,:54]);obs,_,_,_,info=e.step_candidate(a,np.ones(6),np.ones(6));infos.append(info)
            rows.append(dict(date=e.data.scenario_names[day],ac_cost=sum(i['ac_cost']+i['terminal_penalty'] for i in infos),
                ac_violations=sum(i['ac_violations'] for i in infos),constraints=sum(i['constraint_violations'] for i in infos),
                rmse=float(np.sqrt(np.mean([i['research_dt_mse'] for i in infos]))),
                simultaneous_day_covered=all(i['interval_covered'] for i in infos),
                guard_feasible_steps=sum(i['guard_feasible'] for i in infos),guard_certified_steps=sum(i['guard_certificate_survived'] for i in infos)))
        results[name]=dict(estimation=assess(model,records),closed_loop=rows,halfwidth=model['halfwidth'])
        print('DT验证完成：'+name,flush=True)
    results['paired_cost_delta_residual_minus_physics']=[r['ac_cost']-p['ac_cost'] for r,p in zip(results['residual']['closed_loop'],results['physics']['closed_loop'])]
    results['scope']='独立真实日期、相同MPC和执行器；分布变化下报告覆盖实测，非名义保证；未用测试日期选择模型'
    (root/'dt_validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--days',type=int,default=3)
    a=p.parse_args();run(a.folder,days=a.days)
