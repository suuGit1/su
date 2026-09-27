"""统一审计并绘制训练、预算、安全与典型日图表。"""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vpp_research.report_v3 import report_v3


def build(folder):
    root=Path(folder);audit=report_v3(root);out=root/'figures'
    data=json.loads((root/'results.json').read_text());rows=data['entries']
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for i,e in enumerate(rows):
        results=e.get('results',[]);n=sum(r.get('env_steps',0) for r in results)
        label=f"{e['method']}:{e['seed']}"
        values=[100*sum(r.get('guard_certified_steps',0) for r in results)/n if n else np.nan,
                sum(r.get('violations',0)+r.get('ac_violations',0) for r in results),
                sum(r.get('reserve_invalid',0) for r in results)]
        for j in range(3):axes[j].bar(i,values[j]);axes[j].set_xticks(range(len(rows)),[f"{e['method']}:{e['seed']}" for e in rows],rotation=75)
    for ax,label in zip(axes,['Certified steps (%)','Constraint + AC violations','Unconfirmed reserve steps']):ax.set_ylabel(label)
    fig.savefig(out/'safety.png',dpi=170);fig.savefig(out/'safety.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for p in sorted(root.rglob('training.csv')):
        hist=list(csv.DictReader(p.open()));label=str(p.parent.relative_to(root))
        if not hist:continue
        x=[int(r['env_steps']) for r in hist];v=np.array([json.loads(r['vector']) for r in hist])
        for j in range(3):axes[j].plot(x,v[:,j],alpha=.65,label=label)
    for j,ax in enumerate(axes):ax.set_xlabel('Committed training steps');ax.set_ylabel(['Negative cost / 100','Negative carbon / 100','Reserve / 100'][j])
    fig.suptitle('Training returns (preferences and scenarios vary; not a convergence test)')
    fig.savefig(out/'training.png',dpi=170);fig.savefig(out/'training.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    x=np.arange(len(rows));labels=[f"{e['method']}:{e['seed']}" for e in rows]
    ax.bar(x-.2,[e.get('training_steps',0) for e in rows],width=.4,label='Nominal completed training')
    ax.bar(x+.2,[e.get('actual_training_steps',0) or 0 for e in rows],width=.4,label='Successful training calls')
    ax.set_xticks(x,labels,rotation=60);ax.set_ylabel('Steps');ax.legend()
    fig.savefig(out/'budget.png',dpi=170);fig.savefig(out/'budget.svg');plt.close(fig)
    # 每个方法选择第一个有效测试场景展示，场景日期与偏好写入标题。
    for e in rows:
        r=next((r for r in e.get('results',[]) if not r.get('failed') and r.get('trace_path')),None)
        if r is None:continue
        trajectory=[json.loads(s) for s in Path(r['trace_path']).read_text().splitlines() if s.strip()]
        if not trajectory:continue
        info=[r['feedback'] for r in trajectory];x=range(len(info))
        fig,axes=plt.subplots(3,1,figsize=(9,9),layout='constrained')
        for key,label in [('ac_grid_kw','PCC'),('ess_power_kw','ESS'),('pv_used_kw','PV'),('wind_used_kw','Wind')]:axes[0].plot(x,[i[key] for i in info],label=label)
        axes[0].set_ylabel('Power (kW)');axes[0].legend()
        axes[1].plot(x,[i['ess_soc'] for i in info],label='ESS SOC');axes[1].set_ylabel('SOC');axes[1].legend()
        for key,label in [('issued_energy_candidate_kw','Issued ESS'),('received_energy_candidate_kw','Received ESS'),('interval_guard_candidate_kw','Guarded ESS')]:
            if all(key in i for i in info):axes[2].plot(x,[i[key][0] for i in info],label=label)
        axes[2].plot(x,[i['ess_power_kw'] for i in info],label='Executed ESS');axes[2].set_ylabel('ESS power (kW)');axes[2].set_xlabel('Dispatch step');axes[2].legend()
        fig.suptitle(f"{e['method']} seed {e['seed']}, preference {r['preference']}")
        stem=out/f"dispatch_{e['method']}_{e['seed']}";fig.savefig(str(stem)+'.png',dpi=150);fig.savefig(str(stem)+'.svg');plt.close(fig)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();build(a.folder)
