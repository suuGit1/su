"""严格HV口径保持不变，补全日费用、失败分类、动作偏差和验证预算。"""
import csv
import json
from pathlib import Path
import numpy as np
from .suite import feasible
from .evidence_v3 import audit_trace
from .ledger import Ledger,completed_evidence


def failure_reasons(row):
    reasons=['run_failed'] if row.get('failed') else []
    for k in ('violations','ac_violations','ac_failed','reserve_invalid'):
        if row.get(k,0):reasons.append(k)
    if row.get('ev_unmet_kwh',0)>=1e-5:reasons.append('ev_unmet')
    if abs(row.get('dr_backlog_kwh',0))>=1e-5:reasons.append('dr_backlog')
    return reasons


def report_v4(folder):
    root=Path(folder);data=json.loads((root/'results.json').read_text(encoding='utf-8'));plan=data['manifest']
    days=[];summary=[];learning=[];audits=[]
    for e in data['entries']:
        daily=[]
        for r in e.get('results',[]):
            row=dict(method=e['method'],seed=e['seed'],scenario_seed=r.get('seed'),scenario_date=r.get('scenario_date'),
                preference=json.dumps(r['preference']),failed=r.get('failed',False),env_steps=r.get('env_steps',0),
                feasible=feasible(r),failure_reasons=';'.join(failure_reasons(r)))
            if r.get('vector') is not None:row.update(cost_objective=-100*r['vector'][0],carbon_kg=-100*r['vector'][1],reserve_kwh=100*r['vector'][2])
            path=Path(r['trace_path']) if r.get('trace_path') else None
            infos=[json.loads(s)['feedback'] for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path and path.exists() else []
            row['trace_available']=bool(infos)
            if infos:
                row.update(ac_operating_cost=sum(i['ac_cost'] for i in infos),terminal_penalty=sum(i['terminal_penalty'] for i in infos),
                    cyber_cost=sum(i['cyber_cost'] for i in infos),curtailment_kwh=sum(i['pv_curtail_kw']+i['wind_curtail_kw'] for i in infos)*plan['config']['dt_hours'],
                    certified_steps=sum(i['guard_certificate_survived'] for i in infos),reserve_unconfirmed_steps=sum(not i['reserve_valid'] for i in infos),
                    interval_covered_steps=sum(i['interval_covered'] for i in infos),dt_rmse=float(np.sqrt(np.mean([i['research_dt_mse'] for i in infos]))),
                    ess_candidate_execution_l1_kw=sum(abs(i['received_energy_candidate_kw'][0]-i['ess_power_kw']) for i in infos),
                    ess_zero_execution_with_request_steps=sum(abs(i['received_energy_candidate_kw'][0])>1e-3 and abs(i['ess_power_kw'])<1e-6 for i in infos),
                    emergency_steps=sum(i.get('emergency',False) for i in infos),soc_min_observed=min(i['ess_soc'] for i in infos),soc_terminal=infos[-1]['ess_soc'])
                row['cost_reconciles']=abs(row.get('cost_objective',0)-row['ac_operating_cost']-row['terminal_penalty'])<1e-4
            daily.append(row);days.append(row)
        summary.append(dict(method=e['method'],seed=e['seed'],total_test_cases=len(daily),planned_test_cases=plan['eval_days']*len(plan['preferences']),
            completed_cases=sum(not r['failed'] and r['env_steps']==plan['config']['horizon'] for r in daily),feasible_cases=sum(r['feasible'] for r in daily),
            failure_counts={k:sum(k in r['failure_reasons'].split(';') for r in daily) for k in ('run_failed','violations','ac_violations','ac_failed','reserve_invalid','ev_unmet','dr_backlog')},
            training_steps=e.get('actual_training_steps'),selection_steps=e.get('selection_steps',0)))
    for p in sorted(root.rglob('validation_curve.json')):
        for point in json.loads(p.read_text(encoding='utf-8')):
            learning.append(dict(model=str(p.parent.relative_to(root)),**{k:v for k,v in point.items() if k!='results'}))
            for row in point['results']:audits.append(audit_trace(row['trace_path'],row))
    budget=dict(attempted_calls=0,successful_calls=0,error_calls=0,unknown_calls=0)
    for p in root.rglob('evaluation.sqlite'):
        if 'learning_validation' not in p.parts:continue
        snap=completed_evidence(p.parent)
        if snap:counts=snap['summary']
        else:
            ledger=Ledger(p);counts=ledger.summary();ledger.close()
        for k in budget:budget[k]+=counts[k]
    base=json.loads((root/'v3_report.json').read_text(encoding='utf-8')) if (root/'v3_report.json').exists() else {}
    result=dict(version='4.1.0' if plan['config'].get('safety_revision')==1 else '4.0.0',entries=summary,learning_validation=learning,learning_validation_actual_budget=budget,
        learning_validation_audit=dict(consistent=all(a['consistent'] for a in audits),traces=audits),base_evidence_complete=base.get('evidence_complete'),
        evidence_complete=bool(base.get('evidence_complete')) and all(a['consistent'] for a in audits),
        note='严格HV规则不变；逐日结果包括明确标注的不可行结果；验证曲线不是独立测试；证据完整不代表方法优越。')
    (root/'v4_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for row in days for k in row))
    with (root/'daily_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(days)
    plot(root,days,learning,summary)
    return result


def plot(root,days,learning,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'figures';out.mkdir(exist_ok=True);methods=sorted({r['method'] for r in summary})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for j,key in enumerate(('ac_operating_cost','terminal_penalty','curtailment_kwh')):
        for k,m in enumerate(methods):
            v=[r[key] for r in days if r['method']==m and key in r];axes[j].scatter([k]*len(v),v,s=8,alpha=.3)
        axes[j].set_xticks(range(len(methods)),methods,rotation=45);axes[j].set_ylabel(key)
    fig.suptitle('All recorded cases; paired dates/preferences are not independent replicates');save(fig,'v4_cost_components')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for k,m in enumerate(methods):
        v=[r['feasible_cases']/max(1,r['planned_test_cases']) for r in summary if r['method']==m];axes[0].scatter([k]*len(v),v)
        v=[r['ess_zero_execution_with_request_steps']/max(1,r['env_steps']) for r in days if r['method']==m and 'ess_zero_execution_with_request_steps' in r]
        axes[1].scatter([k]*len(v),v,s=8,alpha=.3)
    for ax in axes:ax.set_xticks(range(len(methods)),methods,rotation=45)
    axes[0].set_ylabel('Feasible cases / planned cases');axes[1].set_ylabel('Nonzero ESS request with zero execution\nFraction of steps');axes[0].set_ylim(-.02,1.02);axes[1].set_ylim(-.02,1.02);save(fig,'v4_feasibility_action_gap')
    if learning:
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for model in sorted({r['model'] for r in learning}):
            rows=sorted([r for r in learning if r['model']==model],key=lambda r:r['env_steps'])
            axes[0].plot([r['env_steps'] for r in rows],[r['hv'] for r in rows],label=model)
            axes[1].plot([r['env_steps'] for r in rows],[r['feasible_days']/r['total_days'] for r in rows])
        axes[0].set_ylabel('Fixed validation HV');axes[1].set_ylabel('Fixed validation feasibility')
        for ax in axes:ax.set_xlabel('Committed training environment steps')
        axes[0].legend(fontsize=6);save(fig,'v4_validation_learning')
