"""全量案例审计与配对机制报告；失败不可静默剔除，日期是主要单位。"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def metrics(rows):
    complete=[r for r in rows if not r['failed'] and r.get('completed')];steps=[s for r in rows for s in r['steps']]
    return dict(requests=len(rows),completed=len(complete),failed=sum(r['failed'] for r in rows),executed_steps=len(steps),
        mean_complete_day_cost=float(np.mean([sum(s['cost'] for s in r['steps']) for r in complete])) if complete else None,
        mean_complete_day_carbon=float(np.mean([sum(s['carbon'] for s in r['steps']) for r in complete])) if complete else None,
        mean_complete_day_reserve=float(np.mean([sum(s['reserve'] for s in r['steps']) for r in complete])) if complete else None,
        rmse=float(np.sqrt(np.mean([s['mse'] for s in steps]))) if steps else None,
        coverage=float(np.mean([s['covered'] for s in steps])) if steps else None,
        all_channel_mean_aoi_seconds=float(np.mean([s['aoi'] for s in steps])) if steps else None,
        certified=sum(s['certificate'] for s in steps),ac_violations=sum(s['ac_violations'] for s in steps),
        invalid_reserve=sum(not s['reserve_valid'] for s in steps),pre_violations=sum(s['pre_violations'] for s in steps),
        interventions=sum(s['intervention']>1e-5 for s in steps),fallbacks=sum(s['fallback'] for s in steps))


def paired(rows,group,left,right,field,condition=None,metric="cost"):
    selected=[r for r in rows if r['design']['group']==group and (condition is None or r['design'].get('condition')==condition)]
    pool={}
    for r in selected:
        d=r['design'];pool[(d['date'],d['seed'],d[field])]=r
    keys=sorted({(date,seed) for date,seed,variant in pool});delta=defaultdict(list);failed=[];missing=[]
    for date,seed in keys:
        pair=[pool.get((date,seed,k)) for k in (left,right)]
        if any(r is None for r in pair):missing.append([date,seed]);continue
        if any(r['failed'] or not r.get('completed') for r in pair):failed.append([date,seed]);continue
        delta[date].append(sum(s[metric] for s in pair[0]['steps'])-sum(s[metric] for s in pair[1]['steps']))
    # 所有设计种子都成功才纳入该日期均值，避免种子失败后的选择性日期统计。
    good={date:float(np.mean(values)) for date,values in delta.items() if not any(p[0]==date for p in failed+missing)}
    return dict(left=left,right=right,condition=condition,metric=metric,date_differences=good,n_dates=len(good),
                mean=float(np.mean(list(good.values()))) if good else None,failed_pairs=failed,missing_pairs=missing,
                interpretation='负值为左者指标较低；成本/碳越低越好，备用越高越好；仅完整配对日期，失败必须另报；不作独立日期显著性声明')


def report(root):
    root=Path(root);frozen=json.loads((root/'frozen.json').read_text());rows=[json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))]
    designs={d['id']:d for d in frozen['designs']};seen=set();issues=[]
    for r in rows:
        d=r['design'];key=d['id']
        if key in seen or designs.get(key)!=d:raise ValueError('重复或不匹配案例')
        seen.add(key)
        if not r['failed'] and (not r.get('completed') or len(r['steps'])!=24):issues.append([key,'不完整却未标失败'])
        if [s['step'] for s in r['steps']]!=list(range(len(r['steps']))):issues.append([key,'步骤不连续'])
        cap=d.get('cyber',{});bw=cap.get('bandwidth_bps',10000.);cpu=cap.get('cpu_cycles_per_second',1.e7)
        for s in r['steps']:
            if min(s['bw'])<-1e-8 or sum(s['bw'])>bw+1e-6 or min(s['cpu'])<-1e-8 or sum(s['cpu'])>cpu+1e-5:issues.append([key,'资源容量超额'])
        if d.get('fault') in ('resource_shortage','ac_rejection') and r['failed'] and r.get('state_step')!=len(r['steps']):issues.append([key,'拒绝后的状态推进异常'])
    groups=defaultdict(list)
    for r in rows:
        d=r['design'];key=(d['group'],d.get('condition',''),d.get('variant',d.get('fault',d['model'])),d.get('task',''))
        groups['|'.join(key)].append(r)
    summary={k:metrics(v) for k,v in groups.items()}
    comparisons=[]
    for left,right in [('residual_physics','physics_physics'),('residual_residual','physics_residual'),('physics_residual','physics_physics'),('residual_residual','residual_physics')]:
        comparisons.append(paired(rows,'dt_factorial',left,right,'model'))
    for condition in frozen['protocol']['resource_settings']:
        for left in ('communication_age','computation_age','joint_age','joint_schedule','joint_risk'):
            comparisons.append(paired(rows,'c3',left,'active_equal','variant',condition))
    for left in ('economic','carbon','reserve'):comparisons.append(paired(rows,'tasks',left,'balanced','task','combined'))
    result=dict(completed_requests=len(rows),planned_requests=len(designs),all_requests_recorded=seen==set(designs),audit_issues=issues,
                summaries=summary,paired_cost=comparisons,
                paired_task_objectives=[paired(rows,'tasks',task,'balanced','task','combined',metric) for task in ('economic','carbon','reserve') for metric in ('cost','carbon','reserve')],
                limits=['C3是公开信息启发式机制实验，不是学习算法性能证明','AoI统计包含6个通道，停用EV/DR通道可抬高该均值；不能等同有效通道陈旧度',
                        '532请求中资源不足与AC拒绝是预设失效试验，应记录停止而非伪造成功','同中心换宽度组合只报告实测覆盖',
                        '三月开发日期；确认集未解封','所有总体均值须结合完成率与备用有效性阅读'])
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# 第一、二阶段机制研究结果','',f'已记录 {len(rows)}/{len(designs)} 个完整日请求；审计问题 {len(issues)} 项。',
           '所有结果为开发集机制研究，未评估封存确认集。','',
           '| 组别 | 完成/请求 | 完成日均成本 | 步覆盖 | 证书/执行步 | AC违规 | 未确认备用 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'NA' if v is None else f'{v:.3f}'
    for k,v in summary.items():
        lines.append(f'| {k.replace(chr(124), " / ")} | {v["completed"]}/{v["requests"]} | {fmt(v["mean_complete_day_cost"])} | {fmt(v["coverage"])} | {v["certified"]}/{v["executed_steps"]} | {v["ac_violations"]} | {v["invalid_reserve"]} |')
    lines+=['','## 配对成本差：先按网络种子聚合到日期','']
    for p in comparisons:lines.append(f'- {p["condition"] or "DT/任务"} {p["left"]} − {p["right"]}: {fmt(p["mean"])}，{p["n_dates"]}日；失败对{len(p["failed_pairs"])}，缺失对{len(p["missing_pairs"])}。')
    lines+=['','## 边界','']+['- '+v for v in result['limits']]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');a=p.parse_args();r=report(a.folder)
    print(json.dumps({k:r[k] for k in ['completed_requests','planned_requests','all_requests_recorded','audit_issues']},ensure_ascii=False))
