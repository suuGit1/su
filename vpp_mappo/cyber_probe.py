"""固定反馈控制器下的因果探针；用于验证闭环，不作为训练算法性能排名。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .config import Config
from .environment import VPPAdapter
from .runner import write_csv
from .cyber_environment import cyber_totals
from .flex_environment import resource_totals


def run(config,output,seed=100000):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('输出目录非空')
    out.mkdir(parents=True,exist_ok=True)
    totals=[]
    for mode in ('normal','no_upload','no_cpu','hold_dt'):
        cfg=replace(config,cyber_mode='joint',cyber_dt_mode='hold' if mode=='hold_dt' else 'physics')
        cfg.validate();env=VPPAdapter(cfg,cfg.train_csv);obs,_=env.reset(seed);rows=[]
        for step in range(cfg.horizon):
            # 反馈控制只读公开的 DT 观察，不使用 env.core 或日志中的真值。
            x=obs[0];need=max(0,x[3]*1000);deadline=max(1,x[5]*cfg.horizon)
            ev=need/(deadline*cfg.dt_hours)
            desired=x[13]*5000+ev-1500+(x[1]-x[2])*100
            raw=np.zeros((18,1));raw[0]=np.arctanh(np.clip(desired/env.spec.power_max[0],-.95,.95))
            raw[1]=np.arctanh(np.clip(2*ev/env.flex.ev_station_kw-1,-.999,.999))
            raw[3:6]=-10;raw[6:]=1
            if mode=='no_upload':raw[6:12]=-10
            if mode=='no_cpu':raw[12:]=-10
            obs,_,_,_,info=env.step(raw);rows.append(dict(step=step,**info))
        total=dict(mode=mode,cost=sum(r['cost'] for r in rows),objective=sum(r['cost']+r['terminal_penalty'] for r in rows),
            violations=sum(r['constraint_violations'] for r in rows),ac_violations=sum(r['ac_violations'] for r in rows),
            **resource_totals(rows),**cyber_totals(rows))
        totals.append(total);write_csv(out/(mode+'_trajectory.csv'),rows)
    write_csv(out/'summary.csv',totals)
    (out/'summary.json').write_text(json.dumps(dict(seed=seed,purpose='固定控制器的因果验证，非算法排名',results=totals),ensure_ascii=False,indent=2),encoding='utf-8')
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/c3_smoke.json')
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=100000)
    a=p.parse_args();run(Config.load(a.config),a.output,a.seed)

if __name__=='__main__':main()
