"""PCC 跟踪、故障后备与新目标训练的可重现小预算验证，不作为收敛实验。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from vpp_mappo.config import Config
from vpp_mappo.optimization import DispatchInfeasible
from .ac_safety import ACCheckedFlex
from .pcc_reserve import target_plan, activate, VERSION
from .train import train, evaluate
from .ols import run as train_ols, select_member


def run(config,dt_model,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    report=dict(protocol='stage11-functional-validation-v1',objective_version=VERSION,activations=[],policies=[],ols=None,
        warning='有限样本与小预算功能验证；不是收敛、多种子显著性或连续区间安全证明')
    if (out/'results.json').exists():
        report=json.loads((out/'results.json').read_text())
        if report['objective_version']!=VERSION:raise ValueError('恢复实验版本不一致')
    def save(): (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for seed in (1700,1701,1702):
        if sum(r['seed']==seed for r in report['activations'])==10:continue
        e=ACCheckedFlex(config,emergency=True);e.reset(seed);p,_=e.plan()
        audit=e.network.audit(e.row(),p[0]['action']);baseline=audit['ac_grid_kw']
        for duration in (1,2):
            for fraction in (-1.,-.5,0.,.5,1.):
                r=activate(e,baseline,20.,fraction,duration);r.update(seed=seed,declared_kw=20.)
                report['activations'].append(r)
        save()
    for seed in (21,22):
        for method in ('ordinary','pareto'):
            if any(r['seed']==seed and r['method']==method for r in report['policies']):continue
            folder=out/f'{method}_{seed}';checkpoint=folder/'latest.pt';cfg=replace(config,seed=seed,episodes=3,cyber_mode='joint',coordinator_mode='off')
            if not checkpoint.exists():train(cfg,dt_model,folder,method=method,reserve_mode='pcc_checked')
            rows=evaluate(checkpoint,[[1.,0.,0.]] if method=='ordinary' else [[.2,.3,.5]],[9100,9101],ac_safe=True)
            report['policies'].append(dict(seed=seed,method=method,training_steps=3*config.horizon,results=rows));save()
            print('completed',method,seed,flush=True)
    folder=out/'ols';path=folder/'ols.json'
    if path.exists():ols=json.loads(path.read_text())
    else:ols=train_ols(replace(config,seed=21,cyber_mode='joint',coordinator_mode='off'),dt_model,folder,total_episodes=4,policies=4)
    report['ols']=ols
    if ols['status']=='completed':
        w=[.2,.3,.5];member=select_member(ols,w)
        # 子策略为固定偏好，其动作不依赖测试偏好；报告同时记录选择权重与训练权重。
        report['ols_test']=dict(requested_preference=w,selected_member=member['index'],
            results=evaluate(folder/member['checkpoint'],[member['weight']],[9100,9101],ac_safe=True))
    save();return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dt-model',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(Config.load('configs/research_smoke.json'),json.loads(Path(a.dt_model).read_text()),a.output)
