"""将完成、失败、未运行分开汇总；开发图不冒充确认性结果。"""
import argparse,json,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

def report(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    groups={};rows=[]
    for name in ('safety','dt','coordinator','coordinator_repaired','c3','network'):
        p=root/name/'results.json'
        if not p.exists():groups[name]=dict(status='not_completed');continue
        data=json.loads(p.read_text());groups[name]=dict(status='completed',failed=data['failed'],successful_calls=data['total_successful_calls'])
        for r in data['results']:
            rows.append(dict(group=name,case=r['case']['name'],seed=r['seed'],day=r['day'],failed=r['failed'],steps=r['steps'],cost=r['cost'],carbon_kg=r['carbon_kg'],reserve_kwh=r['reserve_kwh'],rmse=r['rmse'],coverage=r['interval_coverage'],certified_steps=r['guard_certified_steps'],ac_violations=r['ac_violations'],terminal_soc=r['terminal_soc'],terminal_failed=r['terminal_service_failed'],emergency_steps=r['emergency_steps'],decision_p95_seconds=float(np.percentile(r['decision_seconds'],95)) if r['decision_seconds'] else None))
    diagnostic=root/'diagnosis/results.json';diagnosis=[]
    if diagnostic.exists():
        for case in json.loads(diagnostic.read_text()):
            evaluations=case['results'];folder=root/'diagnosis'/case['case'];history=list(csv.DictReader((folder/'training.csv').open()))
            last=history[-1]
            diagnosis.append(dict(case=case['case'],training=case['training'],evaluation_steps=sum(r['env_steps'] for r in evaluations),evaluation_failures=sum(r['failed'] for r in evaluations),terminal_failures=sum(r.get('terminal_service_failed',False) for r in evaluations),
                requested_preferences=[r['preference'] for r in evaluations],evaluations=evaluations,gradient_last_batch={k:float(v) for k,v in last.items() if 'gradient' in k or 'pure_steps' in k},
                note='梯度余弦仅在对应两个pure_steps均大于0时有定义；0占位不是正交证据'))
        versions={r['objective_version'] for c in diagnosis for r in c['evaluations'] if not r['failed']}
        if len(versions)>1:raise ValueError('不同目标协议禁止混合')
    full=root/'seven/results.json';fair=json.loads(full.read_text()) if full.exists() else None
    data_audit=root/'data_audit.json';audit=json.loads(data_audit.read_text()) if data_audit.exists() else None
    result=dict(groups=groups,diagnosis=diagnosis,mechanisms=rows,seven_acceptance=dict(completed=fair.get('completed'),all_successful=fair.get('all_successful'),methods=[dict(method=e['method'],failed=e.get('failed'),budget_complete=e.get('budget_complete'),failed_or_infeasible=e.get('failed_or_infeasible')) for e in fair['entries']]) if fair else None,
        unresolved=['单训练种子开发诊断不足以证明收敛和多目标优势','C3冻结策略消融不能代替各模式独立重训','完整真实EV/DR、目标边缘硬件能耗缺失','IEEE69复用DT区间未重校准','确认性测试集仍未解封'],data_audit=audit)
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if rows:
        with (out/'mechanisms.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# v4.5 开发证据报告','', '代码接口完成不等于论文假设成立；全部失败和负结果保留。','', '|组别|完成状态|失败案例|成功环境调用|','|---|---|---:|---:|']
    for k,v in groups.items():lines.append(f"|{k}|{v['status']}|{v.get('failed','—')}|{v.get('successful_calls','—')}|")
    lines+=['','## 未完成的证据']+['- '+s for s in result['unresolved']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if diagnosis:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for case in diagnosis:
            good=[r for r in case['evaluations'] if not r['failed'] and not r.get('terminal_service_failed',False)]
            points={tuple(r['preference']) for r in good};values=[np.mean([r['vector'] for r in good if tuple(r['preference'])==w],axis=0) for w in sorted(points)]
            if not values:continue
            v=np.asarray(values)*100
            axes[0].scatter(-v[:,0],-v[:,1],label=case['case']);axes[1].scatter(-v[:,0],v[:,2],label=case['case'])
        for ax,y in zip(axes,['Carbon (kg)','Verified reserve (kWh)']):ax.set_xlabel('Total cost');ax.set_ylabel(y);ax.grid(alpha=.2)
        axes[1].legend(fontsize=8);fig.suptitle('Development diagnostics: one training seed; common terminal band')
        fig.savefig(out/'tradeoffs.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(groups=groups,diagnosis_cases=len(diagnosis),fair_available=bool(fair)),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');report(p.parse_args().root)
