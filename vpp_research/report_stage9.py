"""由真实运行文件生成统一参考前沿、验收清单与科学图；不把失败试验排除出报告。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from vpp_mappo.pareto import hypervolume,igd,non_dominated
from .suite import stats


def run(runs,output):
    root=Path(runs);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def read(name):return json.loads((root/name).read_text())
    entries=[]
    for group in ('campaign','ablations'):
        entries+=read(group+'/results.json')['entries']
    union=[p for e in entries if not e['failed'] for p in e['validation_points']]
    reference=[union[i] for i in non_dominated(union)] if union else []
    aggregates={};seeds=sorted({e['seed'] for e in entries})
    for e in entries:
        if not e['failed']:
            e['common_igd']=igd(e['test_points'],reference) if e['test_points'] and reference else None
            e['hv_sensitivity']={str(ref):hypervolume(e['test_points'],ref) for ref in ([-40.,-40.,0.],[-50.,-50.,0.],[-60.,-60.,0.])}
    for family in sorted({e['family'] for e in entries}):
        group=[e for e in entries if e['family']==family];valid=[e for e in group if not e['failed']];deltas=[]
        for e in valid:
            base=next((b for b in entries if b['family']=='ordinary' and b['seed']==e['seed'] and not b['failed']),None)
            if base:deltas.append(e['hv']-base['hv'])
        defined=[e['common_igd'] for e in valid if e['common_igd'] is not None]
        aggregates[family]=dict(completed=len(valid),failed=len(group)-len(valid),hv=stats([e['hv'] for e in valid]) if valid else None,
            igd=stats(defined) if defined else None,undefined_igd=len(valid)-len(defined),
            paired_hv_vs_ordinary=stats(deltas) if deltas else None,evaluation_failures=sum(e['evaluation_failures'] for e in valid))
    mechanisms=read('mechanisms_actions.json');effects={}
    for condition in mechanisms['conditions']:
        candidate=[];executed=[]
        for seed in mechanisms['seeds']:
            group=[r for r in mechanisms['rows'] if r['condition']==condition and r['seed']==seed]
            a=next(r for r in group if r['method']=='physics');b=next(r for r in group if r['method']=='residual')
            if a['failed'] or b['failed']:continue
            for x,y in zip(a['steps'],b['steps']):
                candidate.append(float(np.abs(np.array(x['requested_action'])-y['requested_action']).sum()))
                executed.append(float(np.abs(np.array(x['executed_action'])-y['executed_action']).sum()))
        effects[condition]=dict(mean_candidate_difference_kw=float(np.mean(candidate)),mean_executed_difference_kw=float(np.mean(executed)))
    reserve=read('reserve_activation.json')['rows'];gb=read('gb_study/results.json')
    gb_summary=dict(manifest=gb['manifest'],dt={},policies=[dict(seed=p['seed'],method=p['method'],failed=p['failed'],
        test_failures=sum(x['failed'] for x in p.get('results',[])),test_episodes=len(p.get('results',[])),error=p.get('error')) for p in gb['policies']])
    for method,r in gb['dt'].items():gb_summary['dt'][method]=dict(estimation=r['estimation'],mean_cost=float(np.mean([x['objective'] for x in r['control']])))
    a=gb['dt']['physics']['control'];b=gb['dt']['residual']['control'];gb_summary['paired_residual_cost_delta']=stats([y['objective']-x['objective'] for x,y in zip(a,b)])
    summary=dict(aggregate=aggregates,common_validation_reference=reference,per_seed=[{k:v for k,v in e.items() if k not in ('validation','test','validation_points','test_points')} for e in entries],
        mechanisms=mechanisms['summary'],action_effects=effects,reserve=dict(attempts=len(reserve),failures=sum(r['failed'] for r in reserve),
        immediate_failures=sum(r.get('error','').startswith('指定备用') for r in reserve),later_failures=sum(r.get('error','').startswith('激活后续') for r in reserve)),
        timing=read('timing.json')['summary'],recalibration=read('gb_recalibration.json'),real_data=gb_summary,
        metric_warning='HV/IGD 的备用分量仍是旧线性代理；不是 AC 认证备用前沿，不可据此宣称安全备用优势')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    # 导出直接来自日志的表格，保留无定义项而非填零。
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','completed_seeds','failed_seeds','HV_mean','HV_std','IGD_mean','paired_HV_CI_low','paired_HV_CI_high'])
        for method,r in aggregates.items():
            delta=r['paired_hv_vs_ordinary'];writer.writerow([method,r['completed'],r['failed'],r['hv']['mean'] if r['hv'] else '',r['hv']['std'] if r['hv'] else '',r['igd']['mean'] if r['igd'] else '',*(delta['ci95'] if delta and delta['ci95'] else ['',''])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8));families=list(aggregates)
    means=[aggregates[k]['hv']['mean'] for k in families];errors=[(aggregates[k]['hv']['ci95'][1]-aggregates[k]['hv']['mean']) for k in families]
    ax.errorbar(range(len(families)),means,yerr=errors,fmt='o',capsize=4);ax.set_xticks(range(len(families)),families,rotation=35,ha='right');ax.set_ylabel('HV (linear reserve proxy), mean and 95% CI');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(out/'comparison.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.2))
    for family in ('ordinary','central','joint'):
        series=[]
        for seed in seeds:
            p=root/'campaign'/f'seed_{seed}'/f'{family}_0'/'training.csv'
            with p.open() as f:values=[-json.loads(r['vector'])[0]*100 for r in csv.DictReader(f)]
            series.append(np.convolve(values,np.ones(10)/10,mode='valid'))
        ax.plot(np.arange(10,10+len(series[0])),np.mean(series,axis=0),label=family)
    ax.set_xlabel('Episode');ax.set_ylabel('Economic cost, 10-episode moving mean');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'learning.svg');plt.close(fig)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.runs,a.output)
