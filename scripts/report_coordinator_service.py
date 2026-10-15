"""独立协调器复验报告：保留全部瓶颈、失败与原版强基线。"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from report_phase12 import metrics


def read(root):
    return [json.loads(p.read_text()) for p in sorted((Path(root)/'cases').glob('*.json'))]


def compare(left,right):
    def key(r):
        d=r['design'];return d['date'],d['seed']
    a={key(r):r for r in left};b={key(r):r for r in right};out={}
    for metric in ('cost','carbon','reserve'):
        bydate=defaultdict(list);bad=[]
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b or any(r['failed'] or not r.get('completed') for r in (a.get(k,{}),b.get(k,{}))):bad.append(k);continue
            if metric=='reserve' and any(not s['reserve_valid'] for r in (a[k],b[k]) for s in r['steps']):bad.append(k);continue
            bydate[k[0]].append(sum(s[metric] for s in a[k]['steps'])-sum(s[metric] for s in b[k]['steps']))
        delta={d:float(np.mean(v)) for d,v in bydate.items() if d not in {k[0] for k in bad}}
        out[metric]=dict(mean=float(np.mean(list(delta.values()))) if delta else None,date_differences=delta,excluded_pairs=bad,n_dates=len(delta))
    return out


def report(root,baseline):
    root=Path(root);rows=read(root);old=read(baseline);f=json.loads((root/'frozen.json').read_text());issues=[]
    expected={d['id']:d for d in f['designs']};seen=set()
    for r in rows:
        d=r['design'];ident=d['id']
        if ident in seen or expected.get(ident)!=d:issues.append([ident,'案例设计不匹配'])
        seen.add(ident)
        if not r['failed'] and (not r.get('completed') or len(r['steps'])!=24):issues.append([ident,'完整性错误'])
        cap=d.get('cyber',{})
        for s in r['steps']:
            for field,limit in [('bw',cap.get('bandwidth_bps',10000)),('cpu',cap.get('cpu_cycles_per_second',1e7))]:
                v=np.array(s[field])
                if not np.isfinite(v).all() or (v<0).any() or v.sum()>limit+1e-5 or abs(v[1])+abs(v[3])>1e-8:issues.append([ident,field+'预算或掩码错误'])
    for path,sha in f['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:issues.append([path,'实现哈希变更'])
    if hashlib.sha256(Path('scripts/run_phase12_service.py').read_bytes()).hexdigest()!=f['runner_sha256']:issues.append(['runner','哈希变更'])
    groups=defaultdict(list)
    for r in rows:
        d=r['design'];groups['/'.join([d['group'],d['condition'],d['revision'],d.get('task','balanced')])].append(r)
    summaries={k:metrics(v) for k,v in groups.items()};contrasts={}
    for k,records in groups.items():
        steps=[s for r in records for s in r['steps']]
        if steps:
            summaries[k]['active_mean_aoi_steps']=float(np.mean([s['active_mean_aoi_steps'] for s in steps]))
            summaries[k]['mean_bandwidth_by_channel']=np.mean([s['bw'] for s in steps],axis=0).tolist()
            summaries[k]['mean_cpu_by_channel']=np.mean([s['cpu'] for s in steps],axis=0).tolist()
    for condition in f['protocol']['resource_settings']:
        for revision in ('service_risk','service_value'):
            left=groups.get(f'c3/{condition}/{revision}/balanced',[])
            for variant in ('uniform','active_equal','joint_risk'):
                right=[r for r in old if r['design']['group']=='c3' and r['design'].get('condition')==condition and r['design'].get('variant')==variant]
                contrasts[f'{condition}/{revision}-v1_{variant}']=compare(left,right)
        contrasts[f'{condition}/service_value-service_risk']=compare(groups.get(f'c3/{condition}/service_value/balanced',[]),groups.get(f'c3/{condition}/service_risk/balanced',[]))
    balanced=[r for r in groups.get('c3/combined/service_value/balanced',[]) if r['design']['seed']==431]
    for task in ('economic','carbon','reserve'):
        contrasts[f'task/{task}-balanced']=compare(groups.get(f'tasks/combined/service_value/{task}',[]),balanced)
    service=dict(fresh_admitted=0,fresh_completed=0,conditional_checks=0,payload_breaches=0,packet_losses=0,overflows=0,telemetry_late_jobs=0)
    for r in rows:
        for step in r['steps']:
            record=step['coordinator'];state=record['service_state'];events=step['cyber_events']
            now=step['step']*state['deadline_seconds'];sampled={e['channel']:e for e in events if e['kind']=='sample'}
            complete={e['channel']:e for e in events if e['kind']=='dt_complete' and abs(e['sampled']-now)<1e-6}
            dropped={e['channel'] for e in events if e['kind'] in ('packet_loss','cpu_overflow','tx_overflow')}
            breached={k for k,e in sampled.items() if e['bits']>state['packet_overhead_bits']+state['payload_bound_bits']+1e-6}
            service['payload_breaches']+=len(breached);service['packet_losses']+=sum(e['kind']=='packet_loss' for e in events)
            service['overflows']+=sum(e['kind'] in ('cpu_overflow','tx_overflow') for e in events)
            service['telemetry_late_jobs']+=step['dt_deadline_misses']
            for k in record['fresh_admitted']:
                service['fresh_admitted']+=1;service['fresh_completed']+=int(k in complete)
                room=state['cpu_count'][k]+state['tx_count'][k]+1<=state['queue_capacity']
                if k in sampled and k not in dropped and k not in breached and room:
                    service['conditional_checks']+=1
                    if k not in complete or complete[k]['time']-now>record['deterministic_completion_bound_seconds']+1e-5:issues.append([r['design']['id'],step['step'],k,'条件服务上界未兑现'])
            if sum(step['cpu'])>state['cpu_cycles_per_second']+1e-5:issues.append([r['design']['id'],'未扣除控制CPU预算'])
    result=dict(service=service,recorded=len(rows),planned=len(expected),all_recorded=seen==set(expected),audit_issues=issues,
                total=metrics(rows),summaries=summaries,contrasts=contrasts,
                baseline_frozen_sha256=hashlib.sha256((Path(baseline)/'frozen.json').read_bytes()).hexdigest())
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# 周期内服务分配复验','',f'记录{len(rows)}/{len(expected)}，审计问题{len(issues)}。仅开发集，未评估封存确认集。','',
           '| 条件/修订/任务 | 完成/请求 | 日均成本 | 日均碳 | 日均备用 | 无效备用步 |','|---|---:|---:|---:|---:|---:|']
    def fmt(x):return 'NA' if x is None else f'{x:.3f}'
    for k,v in summaries.items():lines.append(f'| {k} | {v["completed"]}/{v["requests"]} | {fmt(v["mean_complete_day_cost"])} | {fmt(v["mean_complete_day_carbon"])} | {fmt(v["mean_complete_day_reserve"])} | {v["invalid_reserve"]} |')
    lines+=['','| 配对比较（左减右） | 成本差 | 碳差 | 有效备用差 |','|---|---:|---:|---:|']
    for k,v in contrasts.items():lines.append('| '+k+' | '+' | '.join(fmt(v[m]['mean']) for m in ('cost','carbon','reserve'))+' |')
    lines+=['','所有配对日期、剔除原因与失败计数保存在summary.json。未确认备用的配对不进入有效备用差。成本与碳越低越好，备用越高越好；不得跨指标挑选任务成功标准。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');p.add_argument('--baseline',default='runs/phase12_v1');a=p.parse_args();r=report(a.folder,a.baseline)
    print(json.dumps({k:r[k] for k in ('recorded','planned','all_recorded','audit_issues')},ensure_ascii=False))
