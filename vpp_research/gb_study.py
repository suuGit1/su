"""公开英国能源/碳曲线三目标验证：月份隔离，EV 明确关闭。"""
import argparse
import copy
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.data import CSVProfiles
from .dt import fit,assess,save
from .validate_dt import collect,control
from .train import train,evaluate
from .suite import TEST_WEIGHTS,stats


def run(profiles,output):
    folder=Path(profiles);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    c=Config.load('configs/research_smoke.json');c.horizon=24;c.dt_hours=1.;c.reserve_hours=1.;c.synthetic_carbon_g_per_kwh=None
    sources={k:folder/(v+'.csv') for k,v in [('train','train'),('calibration','validation'),('test','test')]}
    counts={'train':24,'calibration':20,'test':16}
    names={k:CSVProfiles(v,24).scenario_names for k,v in sources.items()}
    if any(len(names[k])<counts[k] for k in counts):raise ValueError('完整日期不足')
    c._ev_bundle=dict(schema_version=1,energy_basis='grid_kwh',source='无 EV 消融；未将美国会话伪装为英国实测',horizon=24,dt_hours=1.,scenarios={n:[] for group in names.values() for n in group})
    datasets_path=out/'dt_datasets.json'
    if datasets_path.exists():datasets=json.loads(datasets_path.read_text())
    else:
        datasets={mode:{k:collect(c,range(30000,30000+counts[k]),mode,v) for k,v in sources.items()} for mode in ('hold','physics')}
        datasets_path.write_text(json.dumps(datasets))
    dt={};models={}
    for method in ('hold','physics','residual'):
        d=datasets['hold' if method=='hold' else 'physics'];model=fit(d['train'],d['calibration'],method);save(model,out/(method+'.json'));models[method]=model
        dt_cache=out/(method+'_evaluation.json')
        if dt_cache.exists():dt[method]=json.loads(dt_cache.read_text())
        else:
            dt[method]=dict(estimation=assess(model,d['test']),control=control(c,model,range(31000,31016),csv_path=sources['test']))
            dt_cache.write_text(json.dumps(dt[method],ensure_ascii=False,indent=2))
        print('DT',method,flush=True)
    policies=[]
    for seed in (11,12,13):
        for method in ('ordinary','pareto'):
            cfg=copy.copy(c);cfg.seed=seed;cfg.episodes=24;cfg.train_csv=str(sources['train']);dest=out/f'{method}_{seed}'
            try:
                if not (dest/'latest.pt').exists():train(cfg,models['residual'],dest,method)
                results=evaluate(dest/'latest.pt',TEST_WEIGHTS if method=='pareto' else [[1.,0.,0.]],range(32000,32016),csv_path=sources['test'],ac_safe=True)
                policies.append(dict(seed=seed,method=method,training_steps=cfg.episodes*24,results=results,failed=False))
            except (RuntimeError,ValueError) as exc:
                policies.append(dict(seed=seed,method=method,failed=True,error=str(exc)))
            (out/'policies.partial.json').write_text(json.dumps(policies,ensure_ascii=False,indent=2))
            print(seed,method,flush=True)
    result=dict(protocol='GB-public-curves-carbon-v1',manifest=json.loads((folder/'manifest.json').read_text()),dt=dt,policies=policies,
        limits=['真实国家能源/碳曲线缩放，非馈线实测','EV 关闭；DR 和设备参数仍是公开声明的假设','碳观测使用上小时实际值，尚无历史发布时点档案','此预算与三种子验证不足以声称收敛或跨季节泛化'])
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profiles',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.profiles,a.output)
