"""统一训练加选择交互预算；独立测试另外计账，不参与模型选择。"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .train import train, evaluate
from .ols import run as train_ols, select_member
from .suite import VALIDATION_WEIGHTS, TEST_WEIGHTS, FIXED_WEIGHTS, feasible, points, REFERENCE
from vpp_mappo.pareto import hypervolume, igd, non_dominated

FAMILIES=('ordinary','fixed','pareto','ols')


def allocation(family, total_episodes, horizon, ols_members=4, validation_scenarios=2):
    if family not in FAMILIES or type(total_episodes) is not int or total_episodes<=0 or type(horizon) is not int or horizon<1 or type(ols_members) is not int or ols_members<3 or type(validation_scenarios) is not int or validation_scenarios<1:
        raise ValueError('方法或总回合预算非法')
    members=ols_members if family=='ols' else 3 if family=='fixed' else 1
    evaluations=4 if family=='pareto' else members
    selection=evaluations*validation_scenarios
    training=total_episodes-selection
    if training<members or training%members:
        raise ValueError('扣除选择交互后须给每个子策略分配相同正整数训练回合')
    return dict(total_steps=total_episodes*horizon,training_steps=training*horizon,
                selection_steps=selection*horizon,members=members,episodes_per_member=training//members)


def run(config, model, output, total_episodes=12, seeds=(31,), families=FAMILIES, ols_members=4):
    budgets={f:allocation(f,total_episodes,config.horizon,ols_members) for f in families}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(protocol='train-plus-selection-budget-v1',objective='pcc_checked',config=config.__dict__,
        model_sha256=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),
        budgets=budgets,seeds=list(seeds),validation_seeds=[8100,8101],test_seeds=[9100,9101],
        validation_preferences=VALIDATION_WEIGHTS,test_preferences=TEST_WEIGHTS)
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('恢复协议不同，拒绝混合实验')
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    entries=[]
    for seed in seeds:
        for family in families:
            dest=out/f'{family}_{seed}';result_path=out/f'{family}_{seed}.json'
            if result_path.exists():entries.append(json.loads(result_path.read_text()));continue
            budget=budgets[family];cfg=replace(config,seed=seed,cyber_mode='joint',coordinator_mode='off')
            result=dict(seed=seed,family=family,failed=False,training_steps=0,selection_steps=0,test_steps=0,
                        budget=budget,budget_matched=False,validation=[],test=[],members=[])
            try:
                if family=='ols':
                    state=train_ols(cfg,model,dest,total_episodes=budget['training_steps']//cfg.horizon,
                                    policies=ols_members,reserve_mode='pcc_checked')
                    result['training_steps']=state['completed_training_steps'];result['selection_steps']=state['completed_selection_steps']
                    if state['status']!='completed':raise RuntimeError('OLS提前停止，未花完预算，不进入同预算比较')
                    for m in state['members']:
                        result['members'].append(dict(checkpoint=str(dest/m['checkpoint']),weight=m['weight'],value=m['validation_value']))
                        result['validation']+=m['validation']
                else:
                    weights=FIXED_WEIGHTS if family=='fixed' else [[1.,0.,0.]]
                    for i,w in enumerate(weights):
                        folder=dest/f'member_{i}';cfg.episodes=budget['episodes_per_member']
                        train(cfg,model,folder,method=family,preference=w,scenario_offset=i*cfg.episodes,reserve_mode='pcc_checked')
                        result['training_steps']+=cfg.episodes*cfg.horizon
                        val=evaluate(folder/'latest.pt',VALIDATION_WEIGHTS if family=='pareto' else [w],[8100,8101],ac_safe=True)
                        result['selection_steps']+=sum(r['env_steps'] for r in val);result['validation']+=val
                        # 单策略Pareto无需依据验证向量选择成员；固定策略须全部场景可行。
                        eligible=all(feasible(r) for r in val)
                        result['members'].append(dict(checkpoint=str(folder/'latest.pt'),weight=w,
                            value=np.mean([r['vector'] for r in val],axis=0).tolist() if eligible else None))
                result['budget_matched']=result['training_steps']+result['selection_steps']==budget['total_steps']
                if not result['budget_matched']:raise RuntimeError('实际训练加选择交互未达到统一预算')
                for w in TEST_WEIGHTS:
                    if family in ('ols','fixed'):
                        valid=[m for m in result['members'] if m['value'] is not None]
                        if not valid:raise RuntimeError('验证集没有可行子策略')
                        member=max(valid,key=lambda m:float(np.dot(w,m['value'])))
                        evaluated_weight=member['weight']
                    else:member=result['members'][0];evaluated_weight=w if family=='pareto' else [1.,0.,0.]
                    # 每个方法均实际运行相同数量测试回合，不缓存重复策略从而隐藏测试交互。
                    rows=evaluate(member['checkpoint'],[evaluated_weight],[9100,9101],ac_safe=True)
                    for r in rows:
                        r['policy_evaluation_preference']=r['preference'];r['preference']=list(w)
                        r['unseen_policy_input_preference']=r.pop('unseen_preference')
                        r['preference_is_selection_weight']=family in ('fixed','ols')
                        r['selected_checkpoint']=member['checkpoint']
                    result['test_steps']+=sum(r['env_steps'] for r in rows);result['test']+=rows
                result['validation_points']=points(result['validation'],2)
                result['test_points']=points(result['test'],2)
                result['test_failed_or_infeasible']=sum(not feasible(r) for r in result['test'])
                result['hv']=hypervolume(result['test_points'],REFERENCE)
            except (RuntimeError,ValueError) as exc:
                result.update(failed=True,error=str(exc))
                # 子策略内部失败时恢复已完成交互，失败回合不会伪装成零成本。
                op=dest/'ols.json'
                if family=='ols' and op.exists():
                    state=json.loads(op.read_text());result['training_steps']=state['completed_training_steps'];result['selection_steps']=state['completed_selection_steps']
                elif family!='ols':
                    failures=list(dest.glob('member_*/failure.json'))
                    if failures:result['training_steps']+=sum(json.loads(p.read_text())['completed_env_steps'] for p in failures)
            result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2));entries.append(result)
            print(seed,family,'matched',result['budget_matched'],'failed',result['failed'],flush=True)
    union=[p for r in entries if not r['failed'] for p in r['validation_points']]
    reference=[union[i] for i in non_dominated(union)] if union else []
    for r in entries:
        r['igd']=igd(r['test_points'],reference) if not r['failed'] and r['test_points'] and reference else None
    report=dict(manifest=manifest,reference_point=REFERENCE,validation_reference=reference,entries=entries,
        limits=['预算是环境交互而不是CPU时间；PCC确认内部求解次数另属计算成本',
                '短预算仅验证流程；失败/预算不足单元保留，不进入优势比较',
                '共同参考来自验证集，不是已知真实前沿；测试不能反向挑选成员'])
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));return report


if __name__=='__main__':
    from vpp_mappo.config import Config
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/research_smoke.json');p.add_argument('--dt-model')
    p.add_argument('--output',required=True);p.add_argument('--total-episodes',type=int,default=12);p.add_argument('--seeds',type=int,nargs='+',default=[31]);p.add_argument('--ols-members',type=int,default=4)
    a=p.parse_args();run(Config.load(a.config),json.loads(Path(a.dt_model).read_text()) if a.dt_model else None,a.output,a.total_episodes,a.seeds,ols_members=a.ols_members)
