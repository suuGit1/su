"""真实日期上的模型族；子策略预算合计，选择只使用独立验证日期。"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .train import train, evaluate
from .suite import FIXED_WEIGHTS, VALIDATION_WEIGHTS, feasible
from .ols import next_weight, select_member


def budgets(total, members):
    if total < members: raise ValueError('总训练回合必须不少于子策略数量')
    q,r=divmod(total,members)
    return [q+int(i<r) for i in range(members)]


def learning(cfg,model,dest,method,preferences,selection_csv,test_csv,selection_days,eval_days,resume=False,ols_policies=4):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    count=3 if method=='fixed' else ols_policies if method=='ols' else 1
    allocation=budgets(cfg.episodes,count)
    members=[];values=[];visited=[];validation=[];offset=0
    for i,episodes in enumerate(allocation):
        priority=None
        if method=='ols':
            w,priority=next_weight(values,visited)
            if w is None:break
            w=w.tolist()
        else:w=FIXED_WEIGHTS[i] if method=='fixed' else [1.,0.,0.]
        child=replace(cfg,episodes=episodes);child._ev_bundle=cfg._ev_bundle;child._flex_record=cfg._flex_record
        folder=dest/f'member_{i}' if count>1 else dest
        algorithm='fixed' if method=='ols' else method
        if not (folder/'latest.pt').exists():
            train(child,model,folder,method=algorithm,preference=w,reserve_mode='pcc_checked',scenario_offset=offset,
                  resume=resume and (folder/'resume.pt').exists(),trace_path=folder/'trajectory.jsonl')
        offset+=episodes
        weights=VALIDATION_WEIGHTS if method=='pareto' else [w]
        rows=evaluate(folder/'latest.pt',weights,range(50000,50000+selection_days),csv_path=selection_csv,ev_bundle=cfg._ev_bundle,ac_safe=True)
        validation+=rows
        member=dict(checkpoint=str(folder/'latest.pt'),weight=w,training_steps=episodes*cfg.horizon,
                    selection_steps=sum(r['env_steps'] for r in rows),priority=priority)
        if len(rows)==len(weights)*selection_days and all(feasible(r) for r in rows):
            member['validation_value']=np.mean([r['vector'] for r in rows],axis=0).tolist()
        members.append(member)
        (dest/'members.json').write_text(json.dumps(members,ensure_ascii=False,indent=2))
        if method=='ols':
            # 不把未确认的备用向量用于乐观线性支持搜索。
            if 'validation_value' not in member:break
            values.append(member['validation_value']);visited.append(w)
    results=[]
    selection_failed=count>1 and not any('validation_value' in m for m in members)
    if not selection_failed:
        for w in preferences:
            member=select_member({'members':members},w) if count>1 else members[0]
            results+=evaluate(member['checkpoint'],[w],range(40000,40000+eval_days),csv_path=test_csv,ev_bundle=cfg._ev_bundle,ac_safe=True)
    return dict(training_steps=sum(m['training_steps'] for m in members),
        planned_training_steps=cfg.episodes*cfg.horizon,selection_steps=sum(m['selection_steps'] for m in members),
        budget_complete=offset==cfg.episodes,selection_failed=selection_failed,members=members,validation=validation,
        results=results,test_steps=sum(r['env_steps'] for r in results),
        reserve_unconfirmed_steps=sum(r.get('reserve_invalid',0) for r in results),
        failed_or_infeasible=int(selection_failed)+sum(not feasible(r) for r in results))
