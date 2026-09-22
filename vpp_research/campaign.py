"""可恢复的预算匹配 C3/算法对比；每个实验单元独立保存失败与运行时间。"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from vpp_mappo.config import Config
from vpp_mappo.pareto import hypervolume,igd,non_dominated
from .train import train,evaluate
from .suite import points,stats,FIXED_WEIGHTS,VALIDATION_WEIGHTS,REFERENCE

FAMILIES=('ordinary','central','fixed','control','communication','computation','joint','no_residual','coordinator')
# 分母 7 的内部网格不与预注册固定权重点重合；逐个核查训练偏好。
TEST_WEIGHTS=[[a/7,b/7,(7-a-b)/7] for a in range(1,6) for b in range(1,7-a)]


def run(config,dt_folder,output,episodes=120,seeds=(1,2,3,4,5),families=FAMILIES,reserve_mode="linear"):
    if episodes<3 or episodes%3:raise ValueError('预算必须能被三个固定权重模型均分')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    models={m:json.loads((Path(dt_folder)/(m+'.json')).read_text()) for m in ('physics','residual')}
    manifest=dict(reserve_mode=reserve_mode,protocol='c3-budget-campaign-v2' if reserve_mode=='pcc_checked' else 'c3-budget-campaign-v1',config=config.__dict__,episodes=episodes,seeds=list(seeds),families=list(families),
        dt_hashes={m:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for m,v in models.items()},
        test_weights=TEST_WEIGHTS,validation_seeds=[8100,8101],test_seeds=[9100,9101,9102],
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__)
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text());previous.setdefault('reserve_mode','linear')
        if previous!=manifest:raise ValueError('恢复实验的协议或版本发生改变')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    entries=[]
    for seed in seeds:
        for family in families:
            result_path=out/f'{family}_{seed}.json'
            if result_path.exists():entries.append(json.loads(result_path.read_text()));continue
            started=time.perf_counter();cfg=replace(config,seed=seed,episodes=episodes,cyber_mode='joint',coordinator_mode='off')
            # 所有 C3 四组关闭协调器，避免固定资源被协调器再次改变。协调器单独成对比较。
            if family=='coordinator':cfg.coordinator_mode='schedule'
            elif family=='no_residual':cfg.coordinator_mode='off'
            mode=family if family in ('control','communication','computation','joint') else 'joint'
            method=family if family in ('ordinary','central','fixed') else 'pareto'
            model=models['physics' if family=='no_residual' else 'residual']
            weights=FIXED_WEIGHTS if family=='fixed' else [[1.,0.,0.]]
            row=dict(seed=seed,family=family,method=method,resource_mode=mode,training_env_steps=episodes*config.horizon,failed=False)
            validation=[];test=[]
            try:
                for i,w in enumerate(weights):
                    cfg.episodes=episodes//len(weights);folder=out/f'seed_{seed}'/f'{family}_{i}';checkpoint=folder/'latest.pt'
                    if not checkpoint.exists():train(cfg,model,folder,method,w,scenario_offset=i*cfg.episodes,resource_mode=mode,reserve_mode=reserve_mode)
                    validation+=evaluate(checkpoint,VALIDATION_WEIGHTS if method=='pareto' else [w],[8100,8101],ac_safe=True)
                    test+=evaluate(checkpoint,TEST_WEIGHTS if method=='pareto' else [w],[9100,9101,9102],ac_safe=True)
                row.update(validation=validation,test=test,validation_points=points(validation,2),test_points=points(test,3))
                row['hv']=hypervolume(row['test_points'],REFERENCE)
                row['evaluation_failures']=sum(r['failed'] for r in test)
            except (RuntimeError,ValueError) as exc:row.update(failed=True,error=str(exc))
            row['elapsed_seconds']=time.perf_counter()-started;result_path.write_text(json.dumps(row,ensure_ascii=False,indent=2));entries.append(row)
            print(seed,family,'failed',row['failed'],'seconds',round(row['elapsed_seconds'],1),flush=True)
    union=[p for e in entries if not e['failed'] for p in e['validation_points']]
    reference=[union[i] for i in non_dominated(union)] if union else []
    for e in entries:
        if not e['failed']:e['igd']=igd(e['test_points'],reference) if e['test_points'] and reference else None
    aggregate={}
    for family in families:
        valid=[e for e in entries if e['family']==family and not e['failed']]
        aggregate[family]=dict(completed=len(valid),failed=len(seeds)-len(valid),hv=stats([e['hv'] for e in valid]) if valid else None,
            undefined_igd=sum(e['igd'] is None for e in valid))
    result=dict(manifest=manifest,reference_point=REFERENCE,validation_reference=reference,aggregate=aggregate,entries=entries,
        limits=['增加预算不自动证明收敛，必须检查学习曲线','四种资源模式固定 18 个动作维度；未启用的维度由执行器忽略','pcc_checked 在训练和测试均运行 AC 修正；旧模式仅测试启用' ])
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/research_smoke.json');p.add_argument('--dt-folder',required=True);p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=120);p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5]);p.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES));p.add_argument('--reserve-mode',choices=['linear','ac_checked','pcc_checked'],default='linear');a=p.parse_args();run(Config.load(a.config),a.dt_folder,a.output,a.episodes,a.seeds,a.families,a.reserve_mode)
