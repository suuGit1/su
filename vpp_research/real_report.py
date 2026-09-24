"""统一真实实验报告：空可行前沿显式保留，不以失败回合计算优势。"""
import csv
import json
from pathlib import Path
import numpy as np
from .suite import points,stats
from vpp_mappo.pareto import non_dominated,hypervolume,igd


def report(folder):
    folder=Path(folder);data=json.loads((folder/'results.json').read_text());plan=data['manifest']
    entries=data['entries'];reference=plan.get('hv_reference',[-100.,-200.,0.])
    union=[p for e in entries for p in points(e.get('validation',[]),plan.get('selection_days',1))]
    empirical=[union[i] for i in non_dominated(union)] if union else []
    rows=[]
    for e in entries:
        candidates=points(e.get('results',[]),plan['eval_days']) if not e.get('failed') else []
        front=[candidates[i] for i in non_dominated(candidates)] if candidates else []
        results=e.get('results',[])
        rows.append(dict(seed=e['seed'],method=e['method'],failed=e.get('failed',False),
            selection_failed=e.get('selection_failed',False),failed_or_infeasible=e.get('failed_or_infeasible'),
            training_steps=e.get('training_steps',0),selection_steps=e.get('selection_steps',0),test_steps=e.get('test_steps',0),
            actual_training_steps=e.get('actual_training_steps'),
            budget_complete=e.get('budget_complete',not e.get('failed',False)),
            feasible_points=len(front),hv=hypervolume(front,reference),
            igd=igd(front,empirical) if front and empirical else None,
            guard_infeasible=sum(r.get('guard_infeasible',0) or 0 for r in results),
            guard_certified_steps=sum(r.get('guard_certified_steps',0) for r in results),
            reserve_unconfirmed_steps=e.get('reserve_unconfirmed_steps',0),front=front))
    aggregate={}
    for m in sorted({r['method'] for r in rows}):
        selected=[r for r in rows if r['method']==m]
        distances=[r['igd'] for r in selected if r['igd'] is not None]
        aggregate[m]=dict(hv=stats([r['hv'] for r in selected]),igd=stats(distances) if distances else None,
            undefined_igd=len(selected)-len(distances),failed_runs=sum(r['failed'] for r in selected),
            incomplete_budgets=sum(not r['budget_complete'] for r in selected))
    result=dict(rows=rows,aggregate=aggregate,hv_reference=reference,empirical_validation_reference=empirical,
        totals={k:sum(e.get(k,0) for e in entries) for k in ('training_steps','selection_steps','test_steps')},
        notes=['IGD参考集只来自独立策略选择日期，不是真实Pareto前沿；验证与测试日期不同，IGD含场景差异。',
        '训练预算合计所有子策略；选择交互另列，不能声称开发总交互等预算。',
        '备用未确认或物理/服务约束失败的整组偏好不进入HV/IGD；空集HV=0、IGD=null。',
        '区间证书是单步线性约束证书，不是AC鲁棒保证。'])
    (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    keys=[k for k in rows[0] if k!='front'] if rows else []
    with (folder/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# 统一真实数据实验报告','', '|算法|种子|训练步|选择步|可行前沿点|HV|IGD|备用未确认步|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"|{r['method']}|{r['seed']}|{r['training_steps']}|{r['selection_steps']}|{r['feasible_points']}|{r['hv']:.4g}|{r['igd']}|{r['reserve_unconfirmed_steps']}|" for r in rows]
    lines+=['']+result['notes'];(folder/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');report(p.parse_args().folder)
