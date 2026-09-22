"""69 节点扩展的闭环与模型保存加载验证；不作为长预算性能结论。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from vpp_mappo.config import Config
from .dt_pcc_study import run as dt_run
from .train import train, evaluate


def run(output, dt_folder):
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('输出目录非空；避免覆盖实验')
    out.mkdir(parents=True,exist_ok=True)
    c=Config.load('configs/research_ieee69.json')
    dt_run(c,dt_folder,out/'mpc_dt.json',seeds=(2010,2011))
    model=json.loads((Path(dt_folder)/'residual.json').read_text())
    rows=[]
    for method in ('ordinary','pareto'):
        folder=out/method
        train(replace(c,episodes=1,seed=51,cyber_mode='joint',coordinator_mode='off'),
              model,folder,method=method,reserve_mode='pcc_checked')
        evaluated=evaluate(folder/'latest.pt',[[.2,.3,.5]] if method=='pareto' else [[1.,0.,0.]],
                       [9102],ac_safe=True)
        for row in evaluated:
            row['method']=method
            row['network_model']=c.network_model
        rows+=evaluated
        (out/'learning_smoke.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--dt-folder',required=True)
    a=p.parse_args()
    run(a.output,a.dt_folder)
